# AGENTS.md — ScalableVectorSearch

Entry point for agent context in this repo.

## Required flow
1. Read this file.
2. Read the nearest `AGENTS.md` for touched files.
3. Keep diffs scoped and minimal.
4. Validate with smallest relevant build/test target.
5. If behavior/API changes, update tests/docs in the same PR.

## Mutable details: source of truth only
- Build/options: `CMakeLists.txt`, `cmake/*.cmake`
- CI/required checks: `.github/workflows/`
- Style/lint: `.clang-format`, `.pre-commit-config.yaml`
- Public C++ API: `include/svs/`
- Python package/runtime: `bindings/python/`

Do not duplicate versions, matrices, or flags in instructions.

## Directory map
- `.github/AGENTS.md`
- `benchmark/AGENTS.md`
- `bindings/AGENTS.md`
- `cmake/AGENTS.md`
- `examples/AGENTS.md`
- `include/svs/AGENTS.md`
- `tests/AGENTS.md`
- `tools/AGENTS.md`
- `utils/AGENTS.md`
