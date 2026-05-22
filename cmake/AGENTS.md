# AGENTS.md — cmake/

Build modules, dependency wiring, and feature toggles.

- `CMakeLists.txt` + `cmake/*.cmake` are authoritative.
- Keep option names/defaults stable unless task requires change.
- Prefer additive options over rewrites.
- Validate option/target changes against CI workflows (`.github/workflows/`).

## Intel-specific modules
- **`cmake/mkl.cmake`:** MKL linkage (static vs dynamic threading). Do not hardcode MKL versions. When changing linkage mode, validate threading behavior in tests.
- **`cmake/multi-arch.cmake`:** AVX-512 / SIMD ISA dispatch. Do not hardcode `-march` or ISA flags outside this file. Changes must align with `include/svs/multi-arch/` runtime dispatch code.
- **`cmake/numa.cmake`:** NUMA-aware memory allocation. Respect NUMA topology assumptions in performance-critical code.
- **`cmake/openmp.cmake`:** Threading model. Do not assume specific OpenMP version or runtime without checking source-of-truth.

## Guardrails
- Do not remove optimization flags without justification and benchmark validation.
- Keep CMake minimum version conservative unless a new feature is required across all CI targets.
