# AGENTS.md — cmake/

Build modules, dependency wiring, and feature toggles.

- `CMakeLists.txt` + `cmake/*.cmake` are authoritative.
- Keep option names/defaults stable unless task requires change.
- Prefer additive options over rewrites.
- Validate option/target changes against CI workflows (`.github/workflows/`).
- Use existing option/target patterns; avoid introducing parallel build paths. Before adding a compile flag or option, check whether an existing one already covers the condition; where the same block genuinely exists in more than one file, either change every copy or factor it into one included file.
- Keep configuration values referenced from existing CMake modules.
- Return a computed value another module consumes from a `function()` via `PARENT_SCOPE`, or publish it as a cache entry; a bare `set()` at file scope is not an interface, and a later module setting the same name collides with it silently.
- Do not hardcode versions/toolchain assumptions in instructions or comments.

## Intel-specific modules
- **`cmake/mkl.cmake`:** MKL linkage (static vs dynamic threading). Do not hardcode MKL versions. When changing linkage mode, validate threading behavior in tests.
- **`cmake/multi-arch.cmake`:** AVX-512 / SIMD ISA dispatch. Do not hardcode `-march` or ISA flags outside this file. Changes must align with `include/svs/multi-arch/` runtime dispatch code.
- **`cmake/numa.cmake`:** NUMA-aware memory allocation. Respect NUMA topology assumptions in performance-critical code.
- **`cmake/openmp.cmake`:** Threading model. Do not assume specific OpenMP version or runtime without checking source-of-truth.

## Guardrails
- Do not remove optimization flags without justification and benchmark validation.
- Keep CMake minimum version conservative unless a new feature is required across all CI targets.
