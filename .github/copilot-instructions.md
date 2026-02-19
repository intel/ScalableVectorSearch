# GitHub Copilot Instructions — ScalableVectorSearch

## Source of truth
This file is the canonical instruction set for Copilot behavior in this repository.
`AGENTS.md` provides project context and tech stack overview.

## Mandatory
1. Read root `AGENTS.md` for project context.
2. Read the nearest directory `AGENTS.md` for edited files.
3. If multiple apply, use the most specific.

## Authoring rules
- Keep suggestions minimal and scoped (do not refactor entire files for 2-line changes).
- Use source-of-truth files for mutable details.
- Do not invent or hardcode versions/flags/matrices.
- Avoid `std::iostream` in performance-critical headers.

## Contribution expectations
- Preserve backward compatibility for public API (`include/svs/`)
- Prefer additive API evolution (avoid duplication of existing functionality)
- Pair public API changes with tests and documentation
- For bug fixes: add regression tests
- Run `pre-commit run --all-files` before proposing changes

## Source-of-truth files
- Build: `CMakeLists.txt`, `cmake/*.cmake` (including `cmake/mkl.cmake`, `cmake/multi-arch.cmake`, `cmake/numa.cmake`)
- Dependencies: `bindings/python/pyproject.toml`, `bindings/python/setup.py`
- CI: `.github/workflows/`
- Style: `.clang-format`, `.pre-commit-config.yaml`
- API: `include/svs/`, `bindings/python/`
- Docs: `docs/` (if present)
