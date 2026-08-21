# Issue for: lisa-analysis-tools/LISAanalysistools (or wherever lisaanalysistools issues are tracked)

**Title:** CPU orbit backend (`pycppdetector`) uses AVX-512BW/VL instructions in 1.2.8, SIGILLs on non-AVX-512 CPUs (regression vs 1.2.3-1.2.7)

---

## Summary

Since `1.2.8`, the Linux x86_64 PyPI wheel's compiled CPU backend for orbit computation
(`lisatools_backend_cpu.pycppdetector`, imported via `lisatools.detector`) contains AVX-512BW/VL
instructions (`vmovdqu8`, `vmovdqu16`). On any CPU that doesn't support AVX-512 (e.g. AMD "Zen 2"
EPYC processors, which never got AVX-512, and plenty of older/smaller Intel parts), importing
`lisatools.detector` — or anything that imports it, including `bbhx.response.fastfdresponse`, and
therefore `bbhx.waveformbuild.BBHWaveformFD` unconditionally — crashes the whole process with
`SIGILL` (`Illegal instruction`, exit code 132). This is not a catchable Python exception, so it
kills a Jupyter kernel, a batch job, anything.

`bbhx` itself is unaffected — its own CPU/CUDA backend extensions have no AVX-512 instructions.
This is specific to `lisaanalysistools`'s compiled backend.

## Reproduction

```python
import lisatools.detector  # SIGILL on a non-AVX-512 CPU, once you get this far
```

Minimal native-level repro, no Python-level lisatools/bbhx logic involved:

```python
import lisatools_backend_cpu.pycppdetector  # SIGILL
```

## Environment where this was hit

- Linux x86_64, AMD EPYC 7662 ("Rome" / Zen 2 — has SSE4.2/AVX/AVX2/FMA, no AVX-512 at all)
- Python 3.12.14
- `lisaanalysistools==1.2.8` (PyPI wheel, `manylinux_2_27_x86_64.manylinux_2_28_x86_64`)
- Also affects `bbhx==1.2.3` indirectly, since `BBHWaveformFD.__init__` always ends up importing
  `lisatools.detector` regardless of `force_backend` (cpu or any cuda variant)

## Diagnosis

```
$ python -X faulthandler -c "import faulthandler; faulthandler.enable(); import lisatools.detector"
Fatal Python error: Illegal instruction
...
  File ".../lisatools/cutils/__init__.py", line 50 in cpu_methods_loader
  ...
    (importing lisatools_backend_cpu.pycppdetector)

$ objdump -d lisatools_backend_cpu/pycppdetector.cpython-312-x86_64-linux-gnu.so \
    | grep -E 'vmovdqu8|vmovdqu16' | wc -l
# non-zero in 1.2.8; zero in 1.2.3, 1.2.4, 1.2.6, 1.2.7 (checked each wheel directly)
```

`vmovdqu8`/`vmovdqu16` are AVX-512BW instructions used with AVX-512VL to operate on
XMM/YMM-width registers — so this can show up even in code that never touches a full ZMM
register, which is presumably why it slipped through if it was only tested on an
AVX-512-capable CI/dev machine.

Checked every published Linux x86_64 wheel for `lisaanalysistools`:

| version | `vmovdqu8`/`vmovdqu16` present |
|---|---|
| 1.2.3 | no |
| 1.2.4 | no |
| 1.2.5 | *(no Linux x86_64 wheel published at all)* |
| 1.2.6 | no |
| 1.2.7 | no |
| 1.2.8 | **yes** |

So this looks like a regression introduced specifically in the `1.2.8` build (whatever changed in
the build environment or in `pycppdetector`'s source between `1.2.7` and `1.2.8`), not something
present from the start.

The `lisaanalysistools-cuda12x` plugin package does not have this issue in either `1.2.7` or
`1.2.8` — it's specific to the CPU-backend wheel.

## Workaround

Pin `lisaanalysistools==1.2.7` (or any of `1.2.3`/`1.2.4`/`1.2.6`) instead of taking latest.

## Suggested fix direction

Whatever compiler flags changed for the `1.2.8` release build of the CPU backend (`-march=native`
on an AVX-512-capable CI/build runner would be the classic cause) should be reverted to a portable
baseline for the manylinux wheel — ideally something like `x86-64-v2` or `x86-64-v3` (AVX2, no
AVX-512), since AVX-512 support is far from universal even on current-generation hardware (no AMD
CPU had it until Zen 4 / Genoa, and it's absent from a lot of Intel's mainstream/older lineup too).
If per-CPU dispatch/optimized kernels are wanted, that would need runtime CPU feature detection
rather than a single statically-compiled baseline.

Happy to provide more diagnostic info (disassembly, wheel contents) if useful.
