# AGENTS.md — cmake/

Build modules, dependency wiring, and feature toggles.

- `CMakeLists.txt` + `cmake/*.cmake` are authoritative.
- Keep option names/defaults stable unless task requires change.
- Prefer additive options over rewrites.
- Validate option/target changes against CI workflows (`.github/workflows/`).

## Intel-specific modules
- **`cmake/mkl.cmake`:** MKL linkage (static vs dynamic threading). Do not hardcode MKL versions. When changing linkage mode, validate threading behavior in tests.
- **`cmake/multi-arch.cmake`:** AVX-512 / SIMD ISA dispatch wiring. Changes must align with `include/svs/multi-arch/` runtime dispatch code.
- **`cmake/dispatch-surface.cmake`:** The declared x86 dispatch surface — fixed extents, ISA levels, and their `-march` budgets. Edit it, never the generated `include/svs/core/distance/dispatch_surface.h`, which every configure overwrites. `cmake/dispatch-checks/` holds the ctest checkers that hold the built binary to this declaration.
- **`cmake/numa.cmake`:** NUMA-aware memory allocation. Respect NUMA topology assumptions in performance-critical code.
- **`cmake/openmp.cmake`:** Threading model. Do not assume specific OpenMP version or runtime without checking source-of-truth.

## Guardrails
- Do not remove optimization flags without justification and benchmark validation.
- Keep CMake minimum version conservative unless a new feature is required across all CI targets.
