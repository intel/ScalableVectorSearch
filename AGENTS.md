# AGENTS.md — ScalableVectorSearch

## What this project is
High-performance C++ library for vector similarity search at billion scale. Uses Intel MKL, AVX-512/multi-arch SIMD dispatch, quantization, NUMA-aware memory, OpenMP threading. Python bindings via pybind11. Archetype: **C++** (with Python bindings).

Tech stack: C++20, Intel MKL, OpenMP, pybind11, CMake.
Core principle: **Performance over simplicity** in hot paths.

## How to work
- Backward compatibility is default for public API (`include/svs/`)
- Performance-critical: avoid allocations in hot loops, respect memory alignment
- For SIMD dispatch changes: consult `cmake/multi-arch.cmake` and `include/svs/multi-arch/`
- Python bindings: update `bindings/python/` and ensure GIL release for blocking MKL calls
- Keep suggestions minimal and scoped: do not refactor entire files for 2-line changes.
- Use the source-of-truth files below for mutable details; do not invent or hardcode versions/flags/matrices.
- Add a file only where the nearest `AGENTS.md` purpose line covers its kind and its role; otherwise it belongs in the directory whose purpose line matches.
- A comment states why the code has to be this way and what breaks if it changes, not what the line does; a documentation block on a public entity states that entity's contract. Fix or delete a comment in the same change that alters the code it describes.
- Where code encodes a constraint a reader cannot infer — a platform-specific guard, a derived numeric bound, a warning suppression — comment the reason and what breaks without it. Standard idioms such as a floating-point tolerance need no justification.
- Delete code a change makes dead rather than committing it disabled: commented-out statements, unused variables, branches the change makes unreachable. A conditional-compilation path is not dead merely because no CI job selects it; code deliberately left inert carries a comment saying why.
- A path that cannot satisfy a request fails with a diagnostic naming the unsupported input and the bound or set it violated, rather than substituting a different configuration; a preprocessor or dispatch chain ends in a catch-all that errors instead of assuming the last case holds.
- Compute a value once where every use can see it: an expression identical in both arms of a conditional belongs before the branch, and one that does not depend on the iteration belongs before the loop or per-element closure.
- Derive a computed limit or estimate from the quantity that actually bounds it, not a request-shaped parameter that happens to be in scope or a counter that a second, larger counter can exceed. Where the true bound cannot be computed, document the assumption the estimate makes.
- For bug fixes: add regression tests.
- Run `pre-commit run --all-files` before proposing changes.
- Fetch a dependency from its canonical upstream, not a personal fork or feature branch; where a fork is genuinely still required, name in a comment the upstream PR or issue that will retire it.
- State a policy, requirement, or API description in one place and reference it from the others; a second copy silently drops the parts that matter and diverges from the original.

## Quick start
Build: `cmake -B build && cmake --build build`
Test: `ctest --test-dir build`
Python: `pip install -e bindings/python/`

## Source-of-truth files
- Build: `CMakeLists.txt`, `cmake/*.cmake` (incl. `cmake/mkl.cmake`, `cmake/multi-arch.cmake`, `cmake/numa.cmake`)
- Dependencies: `bindings/python/pyproject.toml`, `bindings/python/setup.py`
- CI: `.github/workflows/`
- Style: `.clang-format`, `.pre-commit-config.yaml`
- API: `include/svs/`, `bindings/python/`

## Directory AGENTS files
- `.github/AGENTS.md` — CI/CD
- `benchmark/AGENTS.md` — performance benchmarks
- `bindings/AGENTS.md` — C++/Python bindings (router)
- `bindings/cpp/AGENTS.md` — C++ bindings
- `bindings/python/AGENTS.md` — Python bindings
- `cmake/AGENTS.md` — build system
- `examples/AGENTS.md` — examples
- `include/svs/AGENTS.md` — public C++ API
- `include/svs/multi-arch/AGENTS.md` — SIMD/ISA dispatch macros
- `include/svs/quantization/AGENTS.md` — quantization
- `tests/AGENTS.md` — test suite
- `tools/AGENTS.md` — tools
- `utils/AGENTS.md` — utilities
