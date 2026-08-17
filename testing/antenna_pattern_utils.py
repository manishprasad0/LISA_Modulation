"""
Utilities for the antenna-pattern / pre-merger sky-localization idea.

Wraps the bbhx waveform generation + STFT envelope extraction from your
notebook into a single function of a parameter dict, plus a null-finder,
so the sensitivity sweeps (Step 1-5 of the plan) can all reuse the same
pipeline instead of copy-pasting the generation/STFT block each time.

Fixes vs. the original notebook code:
  - The min_freq/max_freq band is now actually applied when taking the
    per-time-bin max (previously computed but unused -> envelope was
    taking the max over the FULL 0-Nyquist range, not just the signal band).
  - dist prior/default switched from Mpc to Gpc per your last message
    (default source moved from 2 Gpc -> 20 Gpc; prior range 1-50 Gpc).
  - priors dict keys are now strings and consistently [low, high] pairs
    (previously inc/lam/beta/psi were nested as [(low, high)], which
    would break a plain `low, high = priors[key]` unpack).
"""

import numpy as np
import scipy as sp
import scipy.signal
from lisatools.utils.constants import *
from bbhx.waveformbuild import BBHWaveformFD

# ---------------------------------------------------------------------------
# Fixed observation setup (kept fixed per your call: t_c and Tobs stay fixed
# across all sweeps, so we don't confound sky-position effects with trivial
# shifts of the whole envelope in time).
# ---------------------------------------------------------------------------
DAY_SI = 24 * 3600
MONTH_SI = 30.45428241 * DAY_SI  # matches your original t_months conversion

dt = 5.0
Tobs = YRSID_SI
N = int(Tobs / dt)
Tobs = N * dt
freq = np.fft.rfftfreq(N, dt)
freq[0] = freq[1]

wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False))

MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]

T_C = Tobs - 5 * DAY_SI  # fixed merger time, same as your notebook

# ---------------------------------------------------------------------------
# Default source parameters (your worked example, dist updated to 20 Gpc)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    m1=2e5,
    m2=1e5,
    chi1z=0.0,
    chi2z=0.7,
    dist=20.0 * 1e9 * PC_SI,   # 20 Gpc (was 2 Gpc in the original code)
    phi_ref=1.0,
    inc=2 * np.pi / 3.0,
    lam=0.0,
    beta=0.0,
    psi=3 * np.pi / 8,
)

# dist given in Gpc here (converted to SI inside make_params / compute_stft_envelope)
PRIORS = {
    "m1":       [1e5, 1e6],
    "m2":       [1e5, 1e6],       # remember: bbhx requires m2 <= m1
    "chi1z":    [-1.0, 1.0],
    "chi2z":    [-1.0, 1.0],
    "dist_Gpc": [1.0, 50.0],      # was 1-50 Mpc, switched to Gpc
    "phi_ref":  [0.0, 2 * np.pi],
    "inc":      [0.0, np.pi],
    "lam":      [0.0, 2 * np.pi],
    "beta":     [-np.pi / 2, np.pi / 2],
    "psi":      [0.0, np.pi],
}


def make_params(**overrides):
    """Start from DEFAULT_PARAMS and override any subset of keys.

    Example:
        make_params(psi=0.4)                  # vary psi only
        make_params(lam=1.2, beta=-0.3)        # vary sky position only
    """
    params = DEFAULT_PARAMS.copy()
    params.update(overrides)
    return params


def compute_stft_envelope(
    params,
    nperseg=15000,
    noverlap=0,
    min_freq=3e-5,
    max_freq=0.1,
    t_final=-1,
    modes=None,
):
    """Generate the bbhx waveform for `params` and return (t_months, envelope).

    envelope is the max-normalized, per-time-bin max |STFT| restricted to
    [min_freq, max_freq] -- same quantity as your original max_zxx, but now
    the frequency band is actually applied before taking the max.

    `modes`: override the default mode content (e.g. modes=[(2, 2)] to
    isolate the dominant mode and test whether a feature is a real
    antenna-pattern effect or mode-mixing/interference between harmonics).
    Defaults to the module-level MODES list.
    """
    use_modes = MODES if modes is None else modes
    wave_freq_domain = wave_gen(
        params["m1"], params["m2"], params["chi1z"], params["chi2z"],
        params["dist"], params["phi_ref"], 0.0, params["inc"],
        params["lam"], params["beta"], params["psi"], T_C,
        freqs=freq, modes=use_modes,
        direct=False, fill=True, squeeze=True, length=1024,
    )[0]
    wave_time_domain = np.fft.irfft(wave_freq_domain, axis=-1)

    f, t, Zxx = sp.signal.stft(
        wave_time_domain[1], fs=1 / dt, nperseg=nperseg, noverlap=noverlap
    )

    min_idx = np.searchsorted(f, min_freq)
    max_idx = np.searchsorted(f, max_freq)

    band = np.abs(Zxx[min_idx:max_idx, :])          # <-- the actual fix
    envelope = band.max(axis=0)
    envelope = envelope / envelope.max()

    t_months = t / MONTH_SI

    return t_months[:t_final], envelope[:t_final]


def find_nulls(t_months, envelope, prominence=0.5, min_gap_months=0.5):
    """Locate local minima (nulls) in the envelope.

    Works in log10 space so 'prominence' is roughly in decades of
    amplitude, which is more meaningful than a linear-amplitude prominence
    given how deep these nulls go.

    Returns (null_times_months, null_depths) sorted by time.
    min_gap_months merges minima closer together than this (guards against
    STFT-window artifacts creating spurious double-minima next to a true null).
    """
    log_env = np.log10(envelope + 1e-300)
    inverted = -log_env

    min_distance_bins = max(1, int(min_gap_months / (t_months[1] - t_months[0])))
    peak_idx, _ = sp.signal.find_peaks(
        inverted, prominence=prominence, distance=min_distance_bins
    )

    order = np.argsort(t_months[peak_idx])
    peak_idx = peak_idx[order]

    return t_months[peak_idx], envelope[peak_idx]


def dist_from_gpc(dist_gpc):
    """Convert a distance in Gpc to the SI units bbhx expects."""
    return dist_gpc * 1e9 * PC_SI


# ---------------------------------------------------------------------------
# Step 3/4: one-at-a-time sensitivity sweeps
# ---------------------------------------------------------------------------

# Which DEFAULT_PARAMS key each sweepable name maps to, and how to convert
# a raw prior value into that key's actual (SI/radian) value.
_SWEEP_KEY = {
    "m1":      ("m1",      lambda v: v),
    "m2":      ("m2",      lambda v: v),
    "chi1z":   ("chi1z",   lambda v: v),
    "chi2z":   ("chi2z",   lambda v: v),
    "dist":    ("dist",    dist_from_gpc),   # PRIORS["dist_Gpc"] -> SI dist
    "phi_ref": ("phi_ref", lambda v: v),
    "inc":     ("inc",     lambda v: v),
    "lam":     ("lam",     lambda v: v),
    "beta":    ("beta",    lambda v: v),
    "psi":     ("psi",     lambda v: v),
}

_PRIOR_LOOKUP = {
    "m1": "m1", "m2": "m2", "chi1z": "chi1z", "chi2z": "chi2z",
    "dist": "dist_Gpc", "phi_ref": "phi_ref", "inc": "inc",
    "lam": "lam", "beta": "beta", "psi": "psi",
}


def sweep_parameter(name, n_points=7, base_overrides=None, envelope_kwargs=None,
                     null_kwargs=None):
    """Vary a single parameter over its PRIORS range, holding all others at
    DEFAULT_PARAMS (or base_overrides, if you want a different fixed point --
    e.g. a different (lam, beta) to check robustness, per Step 5).

    Returns a list of dicts, one per sweep point:
        {"sweep_value": <raw prior value>,
         "null_times":  <array of null times in months>,
         "null_depths": <array of null depths (normalized amplitude)>}
    """
    base_overrides = base_overrides or {}
    envelope_kwargs = envelope_kwargs or {}
    null_kwargs = null_kwargs or {}

    if name not in _SWEEP_KEY:
        raise ValueError(f"Unknown sweep parameter '{name}'. Options: {list(_SWEEP_KEY)}")

    param_key, convert = _SWEEP_KEY[name]
    low, high = PRIORS[_PRIOR_LOOKUP[name]]
    raw_values = np.linspace(low, high, n_points)

    results = []
    for raw_val in raw_values:
        params = make_params(**base_overrides)
        params[param_key] = convert(raw_val)
        t_months, env = compute_stft_envelope(params, **envelope_kwargs)
        null_t, null_d = find_nulls(t_months, env, **null_kwargs)
        results.append(dict(sweep_value=raw_val, null_times=null_t, null_depths=null_d,
                             t_months=t_months, envelope=env))
    return results


def summarize_sweep(results, expected_n_nulls=2):
    """Collapse a sweep_parameter() result into two headline numbers:
    the largest null-time shift (months) and largest null-depth change
    (in dex, i.e. log10 units) seen across the sweep, per null index.

    Only compares sweep points where exactly `expected_n_nulls` were found --
    prints a warning if some points found a different count (usually means
    the null-finder needs tuning for that region, or a null genuinely
    vanished/merged -- worth looking at directly, not silently averaging over).
    """
    clean = [r for r in results if len(r["null_times"]) == expected_n_nulls]
    n_bad = len(results) - len(clean)
    if n_bad:
        print(f"[summarize_sweep] {n_bad}/{len(results)} sweep points did not find "
              f"exactly {expected_n_nulls} nulls -- excluded from summary, inspect separately.")

    if len(clean) < 2:
        return dict(max_time_shift_months=np.nan, max_depth_shift_dex=np.nan, n_used=len(clean))

    times = np.array([r["null_times"] for r in clean])     # (n_points, expected_n_nulls)
    depths = np.array([r["null_depths"] for r in clean])

    time_shift = times.max(axis=0) - times.min(axis=0)      # per-null spread, months
    depth_shift = (np.log10(depths.max(axis=0) + 1e-300)
                    - np.log10(depths.min(axis=0) + 1e-300))  # per-null spread, dex

    return dict(
        max_time_shift_months=time_shift.max(),
        max_depth_shift_dex=np.abs(depth_shift).max(),
        n_used=len(clean),
    )


def summarize_sweep_global_min(results):
    """Alternative summary that tracks the single deepest minimum of the
    whole envelope per sweep point, instead of requiring a fixed number of
    matched nulls. Robust to nulls appearing/disappearing/merging across the
    sweep -- which is exactly what happens for wide lam/beta sweeps, where
    `summarize_sweep`'s fixed-null-count matching throws away most points.

    Use this as the primary/headline metric for the bar chart; it's directly
    comparable across every parameter, including lam/beta.
    """
    times, depths = [], []
    for r in results:
        idx = np.argmin(r["envelope"])
        times.append(r["t_months"][idx])
        depths.append(r["envelope"][idx])
    times = np.array(times)
    depths = np.array(depths)

    return dict(
        max_time_shift_months=times.max() - times.min(),
        max_depth_shift_dex=np.abs(
            np.log10(depths.max() + 1e-300) - np.log10(depths.min() + 1e-300)
        ),
        times=times,
        depths=depths,
        n_used=len(results),
    )


def summarize_sweep_matched(results, prominence=0.5, min_gap_months=0.5, ref_idx=None):
    """Match the *set* of nulls at each sweep point against a reference
    sweep point's null set (nearest-neighbor in time), rather than requiring
    a fixed null count or tracking only the current deepest point.

    This is the metric to use once a parameter can produce more than one
    comparably-deep dip (e.g. inc near pi/2): summarize_sweep_global_min
    would report a spurious "shift" whenever the depth ranking between two
    stationary dips flips, since it only ever looks at whichever dip is
    currently deepest. Matching the full set avoids that.

    n_mismatch counts sweep points where the *number* of detected nulls
    differs from the reference -- that's itself informative (a real
    structural change, e.g. lam/beta) and shouldn't be silently absorbed
    into the shift number.
    """
    if ref_idx is None:
        ref_idx = len(results) // 2
    ref_times, ref_depths = find_nulls(
        results[ref_idx]["t_months"], results[ref_idx]["envelope"],
        prominence=prominence, min_gap_months=min_gap_months,
    )

    max_shift = 0.0
    max_depth_shift = 0.0
    n_mismatch = 0

    for r in results:
        t, d = find_nulls(r["t_months"], r["envelope"],
                           prominence=prominence, min_gap_months=min_gap_months)
        if len(t) != len(ref_times):
            n_mismatch += 1
        if len(t) == 0:
            continue
        for rt, rd in zip(ref_times, ref_depths):
            j = np.argmin(np.abs(t - rt))
            max_shift = max(max_shift, abs(t[j] - rt))
            max_depth_shift = max(
                max_depth_shift,
                abs(np.log10(d[j] + 1e-300) - np.log10(rd + 1e-300)),
            )

    return dict(
        max_time_shift_months=max_shift,
        max_depth_shift_dex=max_depth_shift,
        n_mismatch=n_mismatch,
        n_total=len(results),
    )