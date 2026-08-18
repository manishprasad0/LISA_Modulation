"""
Stage 1a dataset generation: fixed masses/spins/distance/phase/t_ref, varying
only (lam, beta, inc, psi). Pipeline per batch: generate frequency-domain TDI
waveforms with bbhx -> irfft to time domain -> STFT -> per-time-bin max
magnitude ("bumps" array) for each of the 3 TDI channels. irfft/STFT use
torch (not numpy/scipy) since the network downstream is torch-based anyway,
and it keeps the door open for running this on GPU on the cluster.

See SUMMARY.md for the full project context.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from lisatools.utils.constants import YRSID_SI
from bbhx.waveformbuild import BBHWaveformFD

from priors import FIXED_PARAMS, sample_free_parameters


DAY_SI = 24 * 3600

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
STFT_WINDOW = torch.hann_window(NPERSEG, dtype=torch.float64)

wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False))

# bbhx keyword name -> config.yaml parameter name (identical except distance/dist)
_BBHX_PARAM_NAMES = {
    "m1": "m1", "m2": "m2", "chi1z": "chi1z", "chi2z": "chi2z",
    "distance": "dist", "phi_ref": "phi_ref",
    "inc": "inc", "lam": "lam", "beta": "beta", "psi": "psi",
}


def generate_waveforms_frequency_domain(free_params):
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


def waveforms_to_time_domain(wave_freq_domain, device="cpu"):
    wave_freq_domain = torch.from_numpy(wave_freq_domain)
    if device == "mps":
        # MPS has no float64/complex128 support at all -- bbhx's native
        # dtype has to be downcast before it can even be moved there.
        wave_freq_domain = wave_freq_domain.to(torch.complex64)
    return torch.fft.irfft(wave_freq_domain.to(device), n=N, dim=-1)


def compute_bumps(wave_time_domain):
    """STFT each (batch, channel) timeseries and take the per-time-bin max
    magnitude over the full frequency range, for all three TDI channels.

    No frequency-band restriction here (unlike the A/E null-finding sweeps
    in testing/antenna_pattern_utils.py): the T channel's dips can fall
    below the band used there.

    Returns bumps with shape (n_batch, 3, n_time_bins).
    """
    n_batch, n_channels, n_time = wave_time_domain.shape
    # dtype must match too, not just device: on MPS the signal is float32
    # (there's no float64 support to fall back to), so a bare .to(device)
    # would leave the window float64 and torch.stft would error.
    window = STFT_WINDOW.to(dtype=wave_time_domain.dtype, device=wave_time_domain.device)

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


def generate_dataset(n_samples, batch_size=8, rng=None, device="cpu"):
    """Sample free parameters and compute the bumps array for each sample.

    Returns (bumps, free_params) where bumps has shape
    (n_samples, 3, n_time_bins) and free_params maps each free-parameter
    name to an array of shape (n_samples,).
    """
    free_params = sample_free_parameters(n_samples, rng=rng)

    bumps_batches = []
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        batch_free_params = {k: v[start:stop] for k, v in free_params.items()}

        wave_freq_domain = generate_waveforms_frequency_domain(batch_free_params)
        wave_time_domain = waveforms_to_time_domain(wave_freq_domain, device=device)
        bumps = compute_bumps(wave_time_domain)
        bumps_batches.append(bumps.cpu().numpy())

    bumps = np.concatenate(bumps_batches, axis=0)
    return bumps, free_params


def _generate_chunk(args):
    n_samples, batch_size, seed = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    return generate_dataset(n_samples, batch_size=batch_size, rng=rng)


def generate_dataset_multiprocess(n_samples, n_workers=6, batch_size=4, rng=None):
    """Same as generate_dataset, but splits the work across n_workers OS
    processes (CPU-only, this laptop's bbhx backend).

    Benchmarked on this machine (Apple M5, 4 performance + 6 efficiency
    cores): 6 worker processes gave the best throughput (~1.4s/sample vs.
    ~1.9s/sample for a single default-threaded process, roughly a 25%
    speedup). 8-10 workers were *worse* than a single process -- each
    sample's full-year time-domain array is large enough (~150MB) that
    this workload is memory-bandwidth-bound, not compute-bound, so piling
    on more processes past ~6 just adds contention. Each worker is capped
    to 1 thread so its own torch/BLAS calls don't oversubscribe on top of
    the process-level parallelism.

    This is a CPU-specific workaround, not something to carry over once
    running on the cluster's GPU bbhx backend -- there, use device="cuda"
    on generate_dataset directly instead.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    rng = np.random.default_rng() if rng is None else rng
    counts = [n_samples // n_workers + (1 if i < n_samples % n_workers else 0) for i in range(n_workers)]
    seeds = rng.integers(0, 2**31 - 1, size=n_workers)
    chunk_args = [(n, min(n, batch_size), int(s)) for n, s in zip(counts, seeds) if n > 0]

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        chunks = list(ex.map(_generate_chunk, chunk_args))

    bumps = np.concatenate([c[0] for c in chunks], axis=0)
    free_params = {
        name: np.concatenate([c[1][name] for c in chunks]) for name in chunks[0][1]
    }
    return bumps, free_params


def save_dataset(path, bumps, free_params):
    np.savez(path, bumps=bumps, **free_params)


if __name__ == "__main__":
    bumps, free_params = generate_dataset(n_samples=16, batch_size=4)
    save_dataset("dataset.npz", bumps, free_params)
    print(f"saved {bumps.shape[0]} samples, bumps shape {bumps.shape} -> dataset.npz")
