# AGENTS.md — ScalableVectorSearch

## What this project is
High-performance C++ library for vector similarity search at billion scale. Uses Intel MKL, AVX-512/multi-arch SIMD dispatch, LVQ quantization, NUMA-aware memory, OpenMP threading. Python bindings via pybind11. Archetype: **C++** (with Python bindings).

Tech stack: C++20, Intel MKL, OpenMP, pybind11, CMake.
Core principle: **Performance over simplicity** in hot paths.

## How to work
- Backward compatibility is default for public API (`include/svs/`)
- Performance-critical: avoid allocations in hot loops, respect memory alignment
- For SIMD dispatch changes: consult `cmake/multi-arch.cmake` and `include/svs/multi-arch/`
- Python bindings: update `bindings/python/` + ensure GIL release for blocking MKL calls

## Quick start
Build: `cmake -B build && cmake --build build`
Test: `ctest --test-dir build`
Python: `pip install -e bindings/python/`

## Directory AGENTS files
- `.github/AGENTS.md` — CI/CD
- `benchmark/AGENTS.md` — performance benchmarks
- `bindings/AGENTS.md` — C++/Python bindings (router)
- `cmake/AGENTS.md` — build system
- `include/svs/AGENTS.md` — public C++ API
- `tests/AGENTS.md` — test suite

For Copilot/agent behavior policy: `.github/copilot-instructions.md`
