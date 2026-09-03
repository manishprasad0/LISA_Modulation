# BBHx + LISAanalysistools on the MPI-IS cluster

GPU-enabled `bbhx` install for LISA MBHB waveforms, verified on an NVIDIA A100.

- **Environment:** `/lustre/home/mprasad/lisa_project/.venv` (Python 3.12.14)
- **Lock file:** `requirements.lock` (50 packages, all from PyPI)
- **Status:** all 5 upstream tests pass on GPU; both primary use cases verified

---

## 1. Verified status

Last full run on `g121` (NVIDIA A100-SXM4-40GB, driver 580.82.07, CUDA 12.4 module):

```
test_direct_likelihood ... ok
test_fast_fd_response  ... ok
test_full_waveform     ... ok
test_het_likelihood    ... ok
test_phenom_hm         ... ok
Ran 5 tests in 16.374s -- OK
```

The two primary use cases, checked together in a single SNR calculation
(`verify_usecase.py`):

```
wave_gen backend: bbhx_cuda12x
waveform type: ndarray | on GPU: True     shape (3, 10000)
max|A| = 2.5815e-17   max|E| = 2.7806e-17
PSD A finite: True  range [3.203e-47, 4.645e-37]
Optimal SNR (A+E, 1e6+5e5 Msun @ 18 Gpc): 2213.6
```

- `BBHWaveformFD(force_backend="gpu")` -> finite A/E/T channels, staying on device
- `lisatools.sensitivity.get_sensitivity` -> finite PSDs

---

## 2. THE CRITICAL PIN: lisaanalysistools must stay at 1.2.0

**Never run `uv pip install -U` in this environment.**

| Package | Version |
| --- | --- |
| `bbhx` / `bbhx-cuda12x` | 1.2.3 |
| **`lisaanalysistools` / `lisaanalysistools-cuda12x`** | **1.2.0 — pinned** |
| `cupy-cuda12x` | 14.2.0 |
| `numpy` / `scipy` / `h5py` | 2.5.2 / 1.18.1 / 3.14.0 |

### Why

`lisaanalysistools` 1.2.1 ported `pycppdetector` from Cython to pybind11 and
dropped the `ptr` property in the process:

- bbhx's `lisaresponse.pyx` does `cdef size_t orbits_in = orbits`, so it needs an
  integer pointer to the C++ `Orbits` object.
- It gets one via `gpubackendtools.pointeradjust.wrapper()`, which does
  `try: arg.ptr / except AttributeError: <pass the object through unchanged>`.
- In 1.2.1+ `Orbits.ptr` raises `AttributeError` (the pybind11 binding exports
  only `.def_readwrite("orbits", ...)`, never `ptr`), so `wrapper` **silently**
  passes a Python object into a `size_t` slot.

The failure is silent at install time and surfaces much later as:

```
TypeError: an integer is required        # response.pyx line 62
```

Version 1.2.0 still uses Cython and has the working accessor:

```cython
def ptr(self) -> long:
    return <uintptr_t>self.g
```

1.2.0 also still has `utils/parallelbase.py` and `gpubackendtools`, so it has the
`.backend` attribute bbhx 1.2.3 requires at `fastfdresponse.py:133`. It is not
too old.

This affects **every** `BBHWaveformFD` call: `__call__` invokes `self.response_gen`
unconditionally (`waveformbuild.py:542`), the response always passes `self.orbits`
(`fastfdresponse.py:472`), and omitting orbits just defaults to
`EqualArmlengthOrbits()`. There is no way to avoid the code path.

### Adding packages safely

```bash
uv pip install --python .venv/bin/python <newpkg>
uv pip freeze --python .venv/bin/python > requirements.lock   # re-freeze after
```

---

## 3. Reusing this install in another folder

### Option A — share this venv (nothing to install)

The venv is location-independent; call it by absolute path from anywhere:

```bash
/lustre/home/mprasad/lisa_project/.venv/bin/python your_script.py
```

or activate it:

```bash
source /lustre/home/mprasad/lisa_project/.venv/bin/activate
```

One environment shared by both projects — upgrading for one changes the other.

### Option B — a separate identical install

```bash
cd /path/to/new_folder
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r /lustre/home/mprasad/lisa_project/requirements.lock
```

Copy `requirements.lock` into the new folder to make it self-contained.

**Do not `cp -r` the `.venv` directory** — absolute paths are baked into
`pyvenv.cfg` and the script shebangs.

---

## 4. Building from scratch (no lock file)

Only if you need to redo this from nothing. Order matters: install bbhx first,
then force `lisaanalysistools` back down to 1.2.0.

```bash
cd /path/to/project
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python bbhx-cuda12x
uv pip install --python .venv/bin/python \
    "lisaanalysistools==1.2.0" "lisaanalysistools-cuda12x==1.2.0"
uv pip freeze --python .venv/bin/python > requirements.lock
```

`bbhx-cuda12x` ships prebuilt wheels, so no `nvcc` is needed at install time and
this works on a login node.

---

## 5. Runtime requirements

Installing is not enough. On a GPU node, before running:

```bash
module load cuda/12.4          # bbhx supports 11.Y.Z and 12.Y.Z ONLY -- never cuda/13.2
export OPENBLAS_NUM_THREADS=1  # raise to match your request_cpus
```

### Why `OPENBLAS_NUM_THREADS`

OpenBLAS defaults to one thread per core (32+ here) and dies against the per-user
process limit:

```
OpenBLAS blas_thread_init: pthread_create failed for thread 11 of 32:
Resource temporarily unavailable
```

The symptom is misleading — it can surface as a `KeyboardInterrupt`-shaped
traceback during `import numpy`. `run_bbhx_tests.sh` also sets `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS`.

CUDA module options on this cluster: `cuda/12.1`, `cuda/12.4`, `cuda/12.9`
(and `cuda/11.x`). `cuda/12.4` is the tested choice.

---

## 6. Submitting jobs (HTCondor)

This cluster uses HTCondor, not Slurm, and requires a bid:

```bash
condor_submit_bid <bid> <file>.sub
```

The bid maps to `priority = bid - 1000`. A small bid of 15 scheduled within
seconds during testing.

```bash
condor_submit_bid 15 bbhx_tests.sub     # full check + upstream test suite
condor_submit_bid 15 usecase.sub        # the two primary use cases only
tail -f logs/bbhx_tests.out
```

**Checking completion:** poll the Condor **log file**, not `condor_q` —
`condor_q` can transiently return empty for a running job and look like
completion. Event `005` is terminated, `009` aborted, `012` held.

```bash
grep -E '^005 \(<clusterid>' logs/bbhx_tests.log
```

Output files may lag a few seconds behind the termination event on lustre.

---

## 7. Files

| File | Purpose |
| --- | --- |
| `requirements.lock` | Exact pinned environment — the source of truth |
| `check_bbhx.py` | Full diagnostic: host, GPU, imports, CuPy, backends, waveform, upstream suite |
| `run_bbhx_tests.sh` | Wrapper: loads CUDA, caps BLAS threads, runs `check_bbhx.py` |
| `bbhx_tests.sub` | Condor submit for the full check (1 GPU, 2 CPUs, 16 GB) |
| `verify_usecase.py` | Focused check: `BBHWaveformFD` on GPU + `get_sensitivity` -> SNR |
| `run_usecase.sh` / `usecase.sub` | Wrapper and submit file for the above |

`check_bbhx.py` isolates each crash-prone check in a subprocess so one failure
reports itself instead of killing the run. It exits 0 only if everything passes.

---

## 8. Gotchas hit during setup

1. **`lisaanalysistools` 1.2.8 wheels crash with SIGILL** (exit 132) on these AMD
   EPYC nodes (75F3 / 7662), on import of `lisatools_backend_cpu.pycppdetector`.
   Not an AVX-512 issue — the binary contains no AVX-512 instructions. Building
   1.2.8 from GitHub source fixed it, but **1.2.0's wheels do not have this
   problem at all**, so the source build is no longer needed.

2. **`pip install --no-binary lisaanalysistools` silently downgrades to 1.0.17.**
   No 1.2.x release ships an sdist, so the resolver walks backwards to the last
   version that has one. To build from source, clone the tag instead — and note
   the repo's `LATW` submodule uses an `git@github.com:` SSH URL, so
   `pip install git+https://...` fails on host-key verification. Use
   `git clone --depth 1 --branch <tag>` (submodules skipped), then install the
   local path.

3. **`python -m bbhx.tests` (as documented in the README) does not work** — the
   package has no `__main__.py`. Use:
   ```bash
   python -m unittest -v bbhx.tests.test_bbhx
   ```

4. **The test suite auto-selects the GPU** whenever `cupy` imports
   (`force_backend = "gpu" if gpu_available else "cpu"`), so it must run on a GPU
   node and requires `lisaanalysistools-cuda12x` — the CUDA backend for lisatools
   is a *separate package* from bbhx's own.
