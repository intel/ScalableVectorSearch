# AGENTS.md — include/svs/

Public C++ headers and API contracts.

- Backward compatibility is default.
- Prefer additive API evolution (avoid duplicating existing functionality).
- Keep headers self-contained and aligned with CMake C++ standard.
- Pair public API changes with tests and docs.

## Performance-critical constraints
- **Memory alignment:** Respect alignment requirements for SIMD operations (use SVS-specific allocators).
- **SIMD dispatch:** Changes to `include/svs/multi-arch/` must align with `cmake/multi-arch.cmake` ISA dispatch logic.
- **Hot paths:** Avoid heap allocations, virtual calls, and `std::iostream` in performance-critical code.
- **MKL integration:** Do not hardcode MKL versions or assume threading model; check `cmake/mkl.cmake`.

## Failure modes to avoid
- Adding template instantiations without checking compile-time impact.
- Changing header-only code without validating ABI compatibility.
