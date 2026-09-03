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

## MPI-IS cluster setup (GPU / CUDA)

A working `bbhx` GPU environment is set up on the MPI-IS cluster (see `~/Documents/Cluster/README.md`
on the local machine for general cluster usage) at `/fast/mprasad/lisa_modulation_test`
(`uv`-managed, `pyproject.toml` + `uv.lock`) — re-confirmed directly on the cluster (`pwd -P` inside
the project dir resolves to `/lustre/fast/fast/mprasad/lisa_modulation_test`; `/home/mprasad` has no
`lisa_modulation_test` dir at all). An earlier note here claimed it lived under home instead of
`/fast`; that was wrong (or the project was moved back since) — always re-verify with `ssh mpi "ls
/fast/mprasad/lisa_modulation_test"` before trusting a path from documentation rather than assuming
either location. Key points if it needs to be rebuilt or extended:

- The cluster's GPU nodes are A100-SXM4-40GB with driver 580.82.07 (CUDA 13.0 capability) — fully
  backward compatible with CUDA 12.x. `bbhx-cuda12x` + `cupy-cuda12x` are installed, matching the
  `module load cuda/12.4` used in the HTCondor wrapper script.
- Getting GPU-backed `BBHWaveformFD` working also requires `lisaanalysistools-cuda12x` (a separate
  PyPI package, not just `lisaanalysistools`) for orbit computation on the GPU. Any `Orbits` object
  passed to `BBHWaveformFD`'s `response_kwargs` must be constructed with the same
  `force_backend=` as the waveform generator itself, or `LISATDIResponse.orbits` raises an
  `AssertionError` (`bbhx/response/fastfdresponse.py`, the `orbits` setter).
- `~/cluster/htcondor/cuda_wrapper.sh` (personalized, on the cluster) prepends this project's
  `.venv/bin` to `PATH` so jobs use it — points at `/fast/mprasad/lisa_modulation_test/.venv/bin`,
  matching the project's actual location (see above); re-checked directly on the cluster.
- The submit/execute nodes are on a shared Lustre filesystem, but there can be a real propagation
  delay between a file write on the login node (e.g. via `scp`) and it being visible/complete on a
  compute node — a job can transiently see a partially-copied file. Don't assume a file is safe to
  use for a job immediately after writing/copying it; a few seconds' buffer avoids flaky failures.

## Known environment issue: cluster CPU-backend crash (AVX-512, `lisaanalysistools==1.2.8`)

`lisaanalysistools`'s compiled CPU orbit backend (`lisatools_backend_cpu.pycppdetector`, imported
by `lisatools.detector` and pulled in unconditionally by `bbhx.response.fastfdresponse`, so it
loads on *any* `BBHWaveformFD` use, GPU included) is built with AVX-512BW/VL instructions
(`vmovdqu8`/`vmovdqu16`) in the `1.2.8` PyPI Linux x86_64 wheel. This SIGILLs (`Illegal instruction`,
exit 132, no Python traceback) on any CPU without AVX-512 — including the cluster's AMD EPYC 7662
(Zen 2) nodes. Versions `1.2.3`–`1.2.4`, `1.2.6`, `1.2.7` do not have this problem (checked via
`objdump -d ... | grep vmovdqu8`); `1.2.5` has no Linux x86_64 wheel at all. Pin
`lisaanalysistools==1.2.7` (and, if using the CUDA backend, `lisaanalysistools-cuda12x==1.2.7`,
which does not have the AVX-512 issue in either 1.2.7 or 1.2.8) rather than taking the latest.
Not yet reported upstream.

## Known environment issue: macOS CPU-backend crash

On macOS arm64, `BBHWaveformFD(force_backend="cpu")` (and anything that triggers the CPU backend
implicitly, e.g. the default constructor) can hard-crash the process with `SIGABRT` — not a
catchable Python exception — when `bbhx`'s and `lisaanalysistools`' independently-built C++
extensions both load their own vendored copy of `libstdc++.6.dylib` into the same process. Full
root-cause analysis, a minimal repro, and a manual `install_name_tool`/`codesign` workaround are in
`GITHUB_ISSUE_bbhx_macos_crash.md`. If waveform generation dies with exit code 134 and no
traceback on macOS, this is almost certainly why — check that issue file before debugging further.
