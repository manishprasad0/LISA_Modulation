"""
GPU (cluster) dataset generation: CUDA equivalent of simulator.py's bumps-array
pipeline. See CLAUDE.md's "MPI-IS cluster setup" section for the environment
this targets (bbhx-cuda12x + cupy-cuda12x + lisaanalysistools-cuda12x,
`module load cuda/12.4`, A100-SXM4-40GB GPUs).

Unlike simulator.py -- written for this MacBook, where bbhx's CPU backend is
the only backend that runs at all (macOS SIGABRT crash, see
GITHUB_ISSUE_bbhx_macos_crash.md) and MPS has no complex128 support for bbhx
itself, only for the downstream irfft/STFT -- on the cluster bbhx runs
natively on CUDA end-to-end. So the whole pipeline (waveform generation ->
irfft -> STFT) runs on one GPU in one process, with bbhx's cupy output handed
to torch via DLPack (zero-copy, no host round trip). That removes the need
for simulator.py's generate_dataset_fast CPU-process/MPS split entirely --
that split only existed to route around bbhx being CPU-only on the Mac.

Not yet run/benchmarked on the cluster -- batch_size in generate_dataset has
no default on purpose; pick it from actual free GPU memory before a real run
(see that function's docstring).
"""

import time

import cupy as cp
import numpy as np
import torch
from lisatools.detector import EqualArmlengthOrbits
from lisatools.utils.constants import YRSID_SI
from bbhx.waveformbuild import BBHWaveformFD

from priors import FIXED_PARAMS, sample_free_parameters


DAY_SI = 24 * 3600
FORCE_BACKEND = "cuda12x"  # matches bbhx-cuda12x / cupy-cuda12x / lisaanalysistools-cuda12x + `module load cuda/12.4`

dt = 5.0
Tobs = YRSID_SI
N = int(Tobs / dt)
Tobs = N * dt
freq = np.fft.rfftfreq(N, dt)
freq[0] = freq[1]  # freq[0] == 0 breaks the waveform generator

MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]

T_C = Tobs - FIXED_PARAMS["t_ref_offset_days"] * DAY_SI

NPERSEG = int(DAY_SI / dt)  # 1 day segments
NOVERLAP = NPERSEG // 2  # 12 hour overlap
HOP_LENGTH = NPERSEG - NOVERLAP
STFT_WINDOW = torch.hann_window(NPERSEG, dtype=torch.float64, device="cuda")

# Orbits must be constructed with the same force_backend as the waveform
# generator, or LISATDIResponse.orbits raises an AssertionError the first
# time a waveform is generated (bbhx/response/fastfdresponse.py's orbits
# setter compares backend names) -- see CLAUDE.md. Left unset, it would
# default to a CPU-backend EqualArmlengthOrbits().
_orbits = EqualArmlengthOrbits(force_backend=FORCE_BACKEND)
wave_gen = BBHWaveformFD(
    amp_phase_kwargs=dict(run_phenomd=False),
    response_kwargs=dict(orbits=_orbits),
    force_backend=FORCE_BACKEND,
)

# bbhx keyword name -> config.yaml parameter name (identical except distance/dist)
_BBHX_PARAM_NAMES = {
    "m1": "m1", "m2": "m2", "chi1z": "chi1z", "chi2z": "chi2z",
    "distance": "dist", "phi_ref": "phi_ref",
    "inc": "inc", "lam": "lam", "beta": "beta", "psi": "psi",
}


def generate_waveforms_frequency_domain(free_params):
    """Same batching/parameter logic as simulator.py's version, but on the
    cuda12x backend wave_gen returns a cupy array (bbhx's `self.xp.zeros`
    buffer is cupy when force_backend is a CUDA backend), not numpy.
    """
    n_batch = len(next(iter(free_params.values())))
    ones = np.ones(n_batch)

    def value_for(param_name):
        if param_name in free_params:
            return free_params[param_name]
        return FIXED_PARAMS[param_name] * ones

    return wave_gen(
        **{kw: value_for(name) for kw, name in _BBHX_PARAM_NAMES.items()},
        f_ref=FIXED_PARAMS["f_ref"],
        t_ref=T_C * ones,
        freqs=freq,
        modes=MODES,
        direct=False,
        length=1024,
        squeeze=False,  # keep the batch axis even when n_batch == 1
        compress=True,
        fill=True,
        combine=False,  # combine=True sums waveforms across the batch into one stream
    )


def waveforms_to_time_domain(wave_freq_domain):
    """cupy (GPU) -> torch (GPU) via DLPack, zero-copy, no host round trip.

    Requires cupy/torch versions that support complex128 over DLPack (cupy
    >=10, torch >=1.11) -- both comfortably satisfied by the cuda12x stack
    pinned on the cluster, but if this ever raises on the torch side, fall
    back to `torch.from_numpy(cp.asnumpy(wave_freq_domain)).to("cuda")`.
    """
    wave_freq_domain = torch.from_dlpack(wave_freq_domain)
    return torch.fft.irfft(wave_freq_domain, n=N, dim=-1)


def compute_bumps(wave_time_domain):
    """Identical to simulator.py's compute_bumps -- torch.stft is backend-
    agnostic and just runs on whatever device wave_time_domain is already on.

    See simulator.py's docstring for why there's no frequency-band
    restriction here (unlike the A/E null-finding sweeps in
    testing/antenna_pattern_utils.py): the T channel's dips can fall below
    the band used there.
    """
    n_batch, n_channels, n_time = wave_time_domain.shape
    window = STFT_WINDOW.to(dtype=wave_time_domain.dtype)

    zxx = torch.stft(
        wave_time_domain.reshape(-1, n_time),
        n_fft=NPERSEG,
        hop_length=HOP_LENGTH,
        win_length=NPERSEG,
        window=window,
        center=True,
        onesided=True,
        return_complex=True,
    )
    zxx = zxx.reshape(n_batch, n_channels, *zxx.shape[-2:])
    return zxx.abs().amax(dim=-2)


def _save_partial(output_path, bumps_batches, free_params, samples_done):
    bumps = np.concatenate(bumps_batches, axis=0)
    partial_free_params = {k: v[:samples_done] for k, v in free_params.items()}
    save_dataset(output_path, bumps, partial_free_params)
    return bumps, partial_free_params


def generate_dataset(n_samples, output_path, batch_size, rng=None, checkpoint_every=50):
    """Single-process, single-GPU generation loop.

    Unlike simulator.py's generate_dataset_fast, there's no CPU-process/MPS
    split to manage: bbhx runs on the GPU directly here, so waveform
    generation, irfft, and STFT all run back-to-back on the same device with
    no host round trip in between (aside from bringing each batch's small
    bumps array back to host for np.savez).

    batch_size has no default -- pick it from the GPU's actual free memory
    before running anything for real. A raw frequency-domain waveform is
    ~150MB/sample (3 channels x ~3.16e6 freq bins x complex128, see
    simulator.py's docstring for the calc) and the time-domain array is
    similar, so on a 40GB A100 something like batch_size 8-16 is a
    reasonable starting point to benchmark from -- this hasn't been
    profiled on the cluster yet.

    Returns (bumps, free_params). Periodically overwrites output_path with
    everything generated so far every checkpoint_every batches, purely so a
    crash doesn't lose the whole run -- like generate_dataset_fast, this is
    NOT a resume mechanism: re-running for the remainder does not pick up
    where a prior run left off.
    """
    rng = np.random.default_rng() if rng is None else rng
    free_params = sample_free_parameters(n_samples, rng=rng)

    bumps_batches = []
    t0 = time.time()
    for batch_idx, start in enumerate(range(0, n_samples, batch_size)):
        stop = min(start + batch_size, n_samples)
        batch_free_params = {k: v[start:stop] for k, v in free_params.items()}

        wave_freq_domain = generate_waveforms_frequency_domain(batch_free_params)
        wave_time_domain = waveforms_to_time_domain(wave_freq_domain)
        bumps = compute_bumps(wave_time_domain)
        bumps_batches.append(bumps.cpu().numpy())
        del wave_freq_domain, wave_time_domain, bumps

        samples_done = stop
        elapsed = time.time() - t0
        rate = samples_done / elapsed
        eta_hours = (n_samples - samples_done) / rate / 3600 if rate > 0 else float("nan")
        print(
            f"batch {batch_idx + 1}: {samples_done}/{n_samples} samples "
            f"({elapsed:.0f}s elapsed, {rate:.3f} samples/s, ETA {eta_hours:.2f}h)",
            flush=True,
        )

        if checkpoint_every and (batch_idx + 1) % checkpoint_every == 0:
            _save_partial(output_path, bumps_batches, free_params, samples_done)

    return _save_partial(output_path, bumps_batches, free_params, n_samples)


def save_dataset(path, bumps, free_params):
    np.savez(path, bumps=bumps, **free_params)


if __name__ == "__main__":
    # Quick sanity check of the GPU path, not the real large run. Must be
    # run on a GPU compute node (e.g. via the HTCondor cuda_wrapper.sh from
    # CLAUDE.md), not on a login node -- login nodes have no GPU.
    bumps, free_params = generate_dataset(n_samples=24, output_path="dataset.npz", batch_size=8)
    print(f"saved {bumps.shape[0]} samples, bumps shape {bumps.shape} -> dataset.npz")
