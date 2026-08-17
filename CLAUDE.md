# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research sandbox for studying antenna-pattern / sky-localization modulation in pre-merger LISA
massive-black-hole-binary signals. It generates frequency-domain waveforms with `bbhx`
(`BBHWaveformFD`), transforms to the time domain, takes an STFT, and studies how the envelope's
nulls (deep dips from LISA's antenna pattern) shift as source parameters are varied.

## Setup / commands

Dependency management is via `uv` (see `pyproject.toml` / `uv.lock`). Requires Python 3.12–3.13,
and is currently pinned to `sys_platform == 'darwin'` in `[tool.uv.environments]`.

```sh
uv sync                 # install/update the environment from uv.lock
uv run jupyter lab       # open the notebooks
uv run python <script>   # run any script inside the project env
```

There is no test suite, lint config, or build step in this repo.

## Code layout

- `antenna_pattern_utils.py` — the reusable pipeline: build source params (`make_params`,
  `DEFAULT_PARAMS`, `PRIORS`), generate a waveform and reduce it to a normalized STFT envelope
  (`compute_stft_envelope`), find envelope nulls (`find_nulls`), and run/summarize one-parameter
  sensitivity sweeps (`sweep_parameter`, `summarize_sweep*`). This module exists specifically so
  sweep notebooks don't re-copy the generate → STFT → envelope block; new sweep-style analysis
  should extend it rather than duplicating waveform-generation code in a notebook.
- `antenna_modulation.ipynb` — main working notebook. Early cells are the original from-scratch
  exploration (waveform generation, raw STFT colormesh plots); later cells (from the "Code from
  Claude" markdown cell onward) use `antenna_pattern_utils` for the parameter sweeps and summary
  plots. When editing, prefer extending the utils module and calling it from new cells rather than
  inlining generation/STFT logic again.
- `bbhx_issue.ipynb` — minimal repro notebook for the CPU-backend crash below.
- `GITHUB_ISSUE_bbhx_macos_crash.md` — drafted upstream bug report (not yet filed) for a SIGABRT
  crash affecting `bbhx` + `lisaanalysistools` on macOS arm64.

## Key domain facts worth knowing before touching the waveform code

- Observation setup (`dt=5s`, `Tobs=YRSID_SI`, `T_C = Tobs - 5 days`) is deliberately held fixed
  across all sweeps in `antenna_pattern_utils.py`, so that varying a source parameter isn't
  confounded with trivially shifting the whole envelope in time.
- `freq[0]` is overwritten to `freq[1]` before being passed to `BBHWaveformFD` — `freq[0] == 0`
  breaks the waveform generator.
- Distances are stored/converted in Gpc (`dist_from_gpc`, `PRIORS["dist_Gpc"]`) but `bbhx` itself
  wants SI distance; `make_params`/`DEFAULT_PARAMS` hold the SI value already.
- `compute_stft_envelope`'s per-time-bin max must be restricted to `[min_freq, max_freq]` before
  taking the max — taking the max over the full 0–Nyquist range silently includes noise/other
  content outside the signal band.
- `find_nulls` operates in log10 space (so `prominence` is roughly in amplitude decades) and merges
  minima closer than `min_gap_months` to avoid STFT-window artifacts creating spurious double
  nulls next to a true one.
- There are three sweep-summary strategies in `antenna_pattern_utils.py` with different tradeoffs
  — pick the one matching what a given parameter's null structure does across its prior range:
  - `summarize_sweep`: fastest, but requires every sweep point to find exactly the same number of
    nulls, or the point is dropped.
  - `summarize_sweep_global_min`: robust to nulls appearing/disappearing/merging; tracks only the
    single deepest minimum. Use as the default/headline metric, except where a parameter can
    produce two comparably deep dips (see below).
  - `summarize_sweep_matched`: matches the full null set against a reference sweep point instead of
    only the deepest point. Needed once a parameter (e.g. `inc` near π/2) can produce two
    comparably-deep dips, where `summarize_sweep_global_min` would report a spurious shift purely
    from the depth ranking between two stationary dips flipping.

## Known environment issue: macOS CPU-backend crash

On macOS arm64, `BBHWaveformFD(force_backend="cpu")` (and anything that triggers the CPU backend
implicitly, e.g. the default constructor) can hard-crash the process with `SIGABRT` — not a
catchable Python exception — when `bbhx`'s and `lisaanalysistools`' independently-built C++
extensions both load their own vendored copy of `libstdc++.6.dylib` into the same process. Full
root-cause analysis, a minimal repro, and a manual `install_name_tool`/`codesign` workaround are in
`GITHUB_ISSUE_bbhx_macos_crash.md`. If waveform generation dies with exit code 134 and no
traceback on macOS, this is almost certainly why — check that issue file before debugging further.
