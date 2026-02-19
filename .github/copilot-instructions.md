# GitHub Copilot Instructions — ScalableVectorSearch

This file is the canonical instruction set for Copilot behavior in this repository
(required flow, precedence, and fallback rules).

`AGENTS.md` is reserved for AGENTS-aware tooling context and should not duplicate
Copilot policy text.

## Mandatory
1. Read root `AGENTS.md` first.
2. Read the nearest directory `AGENTS.md` for edited files.
3. If multiple apply, use the most specific.

## Authoring rules
- Keep suggestions minimal and scoped.
- Use source-of-truth files for mutable details.
- Do not invent or hardcode versions/flags/matrices.

## Source-of-truth files
- `CMakeLists.txt`, `cmake/*.cmake`
- `.github/workflows/`
- `.clang-format`, `.pre-commit-config.yaml`
- `include/svs/`, `bindings/python/`
