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
import shutil
import tempfile
import time
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


def _generate_and_save_chunk(args):
    n_samples, seed, path = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    free_params = sample_free_parameters(n_samples, rng=rng)
    wave_freq_domain = generate_waveforms_frequency_domain(free_params)
    np.save(path, wave_freq_domain)
    return free_params


def _save_partial(output_path, bumps_chunks, free_params_chunks):
    bumps = np.concatenate(bumps_chunks, axis=0)
    free_params = {name: np.concatenate(values) for name, values in free_params_chunks.items()}
    save_dataset(output_path, bumps, free_params)
    return bumps, free_params


def generate_dataset_fast(
    n_samples, output_path, n_workers=6, samples_per_worker=4, device="mps",
    rng=None, checkpoint_every=50,
):
    """Fastest dataset-generation path found by benchmarking (see the
    conversation this was designed in / SUMMARY.md): decouples bbhx's
    CPU-bound waveform generation from irfft/STFT post-processing, so each
    can run at its own best setting instead of fighting each other.

    Each round of n_workers * samples_per_worker samples:
      1. n_workers processes generate that round's waveforms in parallel
         (pure CPU, thread-capped to 1 each) and save them to a scratch dir.
      2. Those raw waveforms are loaded back and run through irfft+STFT on
         `device` (MPS on this MacBook) in one continuous pass, then deleted.

    Benchmarked on this machine (M5, 4 performance + 6 efficiency cores) at
    ~0.49s/sample, vs. ~1.7-2.0s/sample single-process CPU-only -- roughly
    3.5x faster. Splitting into rounds isn't just a knob to tune: a raw
    waveform is ~150MB/sample, so keeping all of n_samples on disk at once
    (e.g. ~15TB for 10^5 samples) isn't possible -- this interleaving keeps
    peak scratch-disk usage to one round's worth.

    This is a single continuous run, not a resumable one: the whole
    n_samples sequence is drawn from one evolving `rng` (freshly seeded if
    not given), so interrupting it and re-running for the remainder does
    NOT pick up where it left off or reproduce a matching prefix.
    checkpoint_every periodically overwrites output_path with everything
    generated so far, purely so a crash doesn't lose the full run -- not a
    resume mechanism.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    rng = np.random.default_rng() if rng is None else rng
    round_size = n_workers * samples_per_worker
    n_rounds = -(-n_samples // round_size)  # ceil division

    scratch_dir = tempfile.mkdtemp(prefix="lisa_waveform_scratch_")
    bumps_chunks = []
    free_params_chunks = {}
    samples_done = 0
    t0 = time.time()

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for round_idx in range(n_rounds):
                n_this_round = min(round_size, n_samples - samples_done)
                counts = [
                    n_this_round // n_workers + (1 if i < n_this_round % n_workers else 0)
                    for i in range(n_workers)
                ]
                seeds = rng.integers(0, 2**31 - 1, size=n_workers)
                paths = [os.path.join(scratch_dir, f"chunk_{i}.npy") for i in range(n_workers)]
                chunk_args = [(n, int(s), p) for n, s, p in zip(counts, seeds, paths) if n > 0]

                worker_free_params = list(ex.map(_generate_and_save_chunk, chunk_args))

                for (_, _, path), free_params in zip(chunk_args, worker_free_params):
                    wave_freq_domain = np.load(path)
                    wave_time_domain = waveforms_to_time_domain(wave_freq_domain, device=device)
                    bumps = compute_bumps(wave_time_domain).cpu().numpy()
                    os.remove(path)

                    bumps_chunks.append(bumps)
                    for name, values in free_params.items():
                        free_params_chunks.setdefault(name, []).append(values)

                samples_done += n_this_round
                elapsed = time.time() - t0
                rate = samples_done / elapsed
                eta_hours = (n_samples - samples_done) / rate / 3600 if rate > 0 else float("nan")
                print(
                    f"round {round_idx + 1}/{n_rounds}: {samples_done}/{n_samples} samples "
                    f"({elapsed:.0f}s elapsed, {rate:.3f} samples/s, ETA {eta_hours:.2f}h)",
                    flush=True,
                )

                if checkpoint_every and (round_idx + 1) % checkpoint_every == 0:
                    _save_partial(output_path, bumps_chunks, free_params_chunks)
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    return _save_partial(output_path, bumps_chunks, free_params_chunks)


def save_dataset(path, bumps, free_params):
    np.savez(path, bumps=bumps, **free_params)


if __name__ == "__main__":
    # Quick sanity check of the fast path, not the real 10^5-sample run --
    # call generate_dataset_fast(100_000, "dataset.npz") directly for that.
    bumps, free_params = generate_dataset_fast(n_samples=24, output_path="dataset.npz")
    print(f"saved {bumps.shape[0]} samples, bumps shape {bumps.shape} -> dataset.npz")
