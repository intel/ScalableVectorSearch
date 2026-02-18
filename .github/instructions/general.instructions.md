---
applyTo: "**"
---

# General Instructions for GitHub Copilot

## Context loading (required)
- Read root `AGENTS.md` first.
- Then read nearest local `AGENTS.md` for touched files.
- If multiple apply, follow the most specific one.

## Long-lived policy
- Keep this file stable and high-level.
- Put mutable details only in source-of-truth files.
- Reference those files instead of duplicating values.

## Source-of-truth
- Build/options: `CMakeLists.txt`, `cmake/*.cmake`
- CI/checks: `.github/workflows/`
- Style/lint: `.clang-format`, `.pre-commit-config.yaml`
- API/package contracts: `include/svs/`, `bindings/python/`

## Change hygiene
- Prefer minimal scoped edits.
- Preserve backward compatibility by default.
- Pair behavior/API changes with tests/docs updates.
