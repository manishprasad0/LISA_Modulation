# Issue for: lisa-analysis-tools/BBHx

**Title:** CPU backend crashes (SIGABRT) on macOS arm64 when used with `lisaanalysistools` — duplicate vendored `libstdc++`

---

## Summary

On macOS (arm64), importing `BBHWaveformFD` and instantiating it with `force_backend="cpu"` aborts the entire Python process with `SIGABRT` (exit code 134) — not a catchable Python exception, so it takes down a Jupyter kernel or any calling process. Happens on a completely clean `pip install bbhx lisaanalysistools` with no other customization.

## Reproduction

```python
from bbhx.waveformbuild import BBHWaveformFD
wave_gen = BBHWaveformFD(force_backend="cpu")  # process aborts here, no traceback
```

Minimal native-level repro (no bbhx Python code needed at all):

```python
import lisatools_backend_cpu.pycppdetector
import bbhx_backend_cpu.response   # aborts here
```

## Environment

- macOS, arm64 (Apple Silicon)
- Python 3.12.13
- `bbhx==1.2.3` (PyPI wheel)
- `lisaanalysistools==1.2.8` (PyPI wheel)
- Installed via `uv`/`pip`, CPU-only (`force_backend="cpu"`)

## Root cause

`bbhx`'s CPU backend (`bbhx_backend_cpu.response`, and other `bbhx_backend_cpu.*` extensions) and `lisaanalysistools`'s CPU backend (`lisatools_backend_cpu.pycppdetector`) are each independently-built C++ extensions whose macOS wheels vendor their **own separate copy** of `libstdc++.6.dylib` (presumably via `delocate` in each project's own wheel-repair step):

```
$ otool -L bbhx_backend_cpu/response.cpython-312-darwin.so
	@loader_path/../bbhx/.dylibs/libstdc++.6.dylib (compatibility version 7.0.0, current version 7.33.0)

$ otool -L lisatools_backend_cpu/pycppdetector.cpython-312-darwin.so
	@loader_path/../lisatools/.dylibs/libstdc++.6.dylib (compatibility version 7.0.0, current version 7.33.0)
```

Both `.dylib` files are (as far as I can tell) the same GCC libstdc++ build — same compatibility version, same current version, nearly identical size — but they are two **distinct files at two distinct paths**, so `dyld` loads them as two separate images in the same process. `BBHWaveformFD.__init__` → `bbhx.response.fastfdresponse` imports `lisatools.detector` (for orbits) *before* the CPU backend extensions get imported, so both vendored libstdc++ copies end up loaded simultaneously — which reliably aborts the process (no Python-catchable exception, no dyld error printed; it's a hard native abort).

I could reproduce this abort with just the two extension modules above, with none of `bbhx`'s or `lisatools`'s Python-level code involved, confirming this is purely a native/packaging issue and not a logic bug.

This is presumably a byproduct of `bbhx` and `lisaanalysistools` being built as independent wheels (each via its own `cibuildwheel`/`delocate` pipeline) that happen to both statically link/vendor GCC's C++ runtime, rather than sharing one. It will affect anyone using `bbhx`'s CPU backend together with `lisaanalysistools` on macOS.

## Workaround

Repointing one of the two extensions' `LC_LOAD_DYLIB` to the other's already-loaded `libstdc++.6.dylib` (so `dyld` reuses a single image instead of loading two) fixes it:

```sh
install_name_tool -change \
  "@loader_path/../lisatools/.dylibs/libstdc++.6.dylib" \
  "@loader_path/../bbhx/.dylibs/libstdc++.6.dylib" \
  lisatools_backend_cpu/pycppdetector.cpython-312-darwin.so
codesign --force --sign - lisatools_backend_cpu/pycppdetector.cpython-312-darwin.so
```

After this, `BBHWaveformFD(force_backend="cpu")` instantiates fine. This is obviously not something end users should have to do by hand on every install, though.

## Suggested fix direction

I don't have visibility into the exact `cibuildwheel`/`delocate` setup for either package's macOS wheels, so I can't propose a concrete PR, but some options that would fix this at the source:
- Have `bbhx`'s macOS wheel build exclude vendoring its own `libstdc++` and instead depend on/reuse the one vendored by `lisaanalysistools` (or vice versa), since these two packages are both part of the LISA Analysis Tools ecosystem and commonly installed together.
- Or have both use a `delocate`/`auditwheel`-style step that de-duplicates identical vendored libraries by content hash rather than by originating package.
- At minimum, documenting this known interaction (and the `install_name_tool` workaround) would save the next person a crashed kernel with zero error message.

Happy to provide more diagnostic info if useful.
