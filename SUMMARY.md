# Project: LISA Antenna Pattern for Sky Localization

## Background / motivation

The space based gravitational wave (GW) detector, LISA measures GWs from multiple types of sources. Out of these, massive black hole binary (MBHB) mergers are expected to be seen over long (pre-merger) inspirals. LISA will perform a cartwheel like rotation around the sun (rotate about its axis and revolve around the sun). Thus, the detector's antenna pattern changes as the constellation orbits the Sun over the course of the year. This produces amplitude modulation in the signal.

While revisiting old master's thesis LISA code, I noticed something. I was working with the time-frequency domain representation using STFT (short-time Fourier transform). For noiseless signals, if one takes the maxima of each time segment of the STFT, these maximas in the TDI A, E and T channels show a pattern of dips and nulls. My hypothesis is that these patterns and the timing of these dips depends only on sky location (lambda, beta) and maybe also the inclination of the binary and the polarization of the GW — not on the masses, spins, or distance of the binary. This suggests the modulation pattern itself encodes sky localization information in a way that's largely decoupled from the "hard" parameters (masses/spins), which are otherwise the dominant target of most GW parameter estimation.

## Core idea

Train a network that learns this mass/spin-independent antenna pattern structure and predicts only the parameters that actually control it: sky location (lambda, beta), and — to be tested — inclination (i) and polarization angle (psi), since it's not yet confirmed whether psi also affects the null locations.

Motivation: real-time electromagnetic (EM) alerts for LISA mergers need sky localization fast, pre-merger, and don't need masses/spins for that purpose. Sky location (lambda, beta) is the parameter pair that's actually critical for the EM alert use case; inclination (i) and polarization (psi) are also being predicted in stage 1a as part of the sanity check, but are not required for the alert itself. If a lightweight network can extract (lambda, beta) from the modulation pattern alone, pre-merger, that's a much cheaper path to an early-warning sky-localization pipeline than full parameter estimation.

A separate (later, not-yet-started) block is planned to predict merger time from pre-merger data, to support the same early-alert use case.

## Staged proof-of-concept plan

1. **Stage 1a (current):** Fixed masses, spins, distance, and phase. Vary only lambda, beta, inclination (i), and polarization (psi). Noiseless, full year-long duration signals. Goal: sanity check — does a network fed only the "bumps" array (the max-per-STFT-segment modulation curve) learn to predict (lambda, beta, i, psi) at all? This stage deliberately avoids realism (no noise, full year of data, only 4 free parameters) to isolate the question of whether the signal is learnable in principle.
2. **Stage 1b:** Add noise and other realism, get it working under that.
3. **Stage 2:** Move to pre-merger-only data; network also learns coalescence time from the tail of observed data.
4. **Stage 3:** Test with shorter-duration data, since year-long signals are unrealistically long/ideal for a real early-alert use case.

We are at the very start of stage 1a.

## Stage 1a specifics

- **Fixed:** masses, spins, distance, phase, time of coalescence (kept fixed for this stage; varying it is deferred to stage 2, where merger time becomes something the network learns from pre-merger data — not a fixed input).
- **Free / targets:** lambda, beta, inclination (i), polarization (psi).
- **Data:** noiseless, full year-long duration, generated with `bbhx`. `bbhx` is a MBHB waveform generator for LISA in the frequency domain. See the code of `bbhx` that you helped me debug last week.
- **Pipeline per sample:** generate waveform in frequency domain → `irfft` to time domain → STFT → take the max per STFT segment ("bumps" array) for all three TDI channels. For the neural network, the plan is to concatenate this summary array of maximas.
- Open performance question: `irfft` and STFT are currently done with numpy/scipy and are part of the per-sample bottleneck. Since we're going to use `torch` for the neural network anyway, it's worth investigating `torch`'s own `irfft` and STFT implementations as a faster (and possibly GPU-compatible) alternative.
- **Model:** deliberately simple to start — for an embedding network, use a block of multiple ResNets and then a normalizing flow block with neural spline flows. We need a flow model since in physics we ultimately want the posteriors for sky localization, not just point estimates
- **Data volume:** starting with a much smaller dataset than originally planned (tens of thousands, not ~1M samples) specifically for this sanity-check stage, since generation is currently the bottleneck and a full-scale dataset isn't needed to answer the stage 1a question.
- Note: `bbhx` supports batched generation, which should be used to speed up dataset generation rather than generating one sample at a time in a loop. Worth looking into the `bbhx` source code directly to see how it parallelizes internally, to understand how best to use it and whether there's additional speedup available beyond just passing batched arrays.
- For now, the plan is to implement it on this Macbook and later on a cluster with GPUs. We will have to make the code GPU compatible for that implementation.
- Since, the training data generation can take quite long, the plan is to first write a code that generates the waveforms and saves the maximas array along with the true values for training the network. 

## Parameter / prior configuration

Plan (suggested by my supervisor Max (Maximilian Dax at MPI-IS)): keep a YAML config file listing the fixed parameters (masses, spins, distance, phase, time of coalescence for this stage) and the priors for the varied parameters (lambda, beta, i, psi), rather than hardcoding these in `priors.py`/`simulator.py`. This keeps the parameter setup readable and easy to change between stages.

## Current repo state

- `CLAUDE.md` — being rewritten (this file will inform it); the auto-generated one from `claude init` picked up irrelevant context from unrelated files, which have been moved to a `testing/` subfolder.
- `priors.py` — currently empty; will define the prior distributions over lambda, beta, i, psi (and hold masses/spins/distance/phase fixed) for stage 1a data generation.
- `simulator.py` — in progress; implement the waveform generation → time domain → STFT → bumps-array → save for training  pipeline described above.
- `GITHUB_ISSUE_bbhx_macos_crash.md` — drafted upstream bug report (not yet filed) for a SIGABRT crash affecting `bbhx` + `lisaanalysistools` on macOS arm64.

## What this file is for

This is background context for Claude Code to understand the project, current stage, and design decisions already made, so it doesn't need to re-derive them or guess from partial code. It should inform `CLAUDE.md`, not replace it — `CLAUDE.md` should stay focused on repo-specific conventions, file structure, and how to run things, while this file carries the "why."