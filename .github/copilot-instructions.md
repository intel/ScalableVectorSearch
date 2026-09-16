# GitHub Copilot Instructions — ScalableVectorSearch

This file is the canonical instruction set for Copilot behavior in this repository
(required flow, precedence, and fallback rules).

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
- Add a file only where the nearest `AGENTS.md` purpose line covers its kind and its role; otherwise it belongs in the directory whose purpose line matches.
- A comment states why the code has to be this way and what breaks if it changes, not what the line does; a documentation block on a public entity states that entity's contract. When a change alters code a comment describes, fix or delete that comment in the same change.
- Where code encodes a constraint a reader cannot infer — a platform-specific guard, a derived numeric bound, a warning suppression — comment the reason and what breaks without it. Standard idioms such as a floating-point tolerance need no justification.
- Delete code a change makes dead rather than committing it disabled: commented-out statements, unused variables, branches the change makes unreachable. A conditional-compilation path is not dead merely because no CI job selects it. Code deliberately left inert carries a comment saying why.
- A path that cannot satisfy a request fails with a diagnostic naming the unsupported input and the bound or set it violated, rather than substituting a different configuration; a preprocessor or dispatch chain ends in a catch-all that errors instead of assuming the last case holds.
- Compute a value once where every use can see it rather than repeating the computation: an expression identical in both arms of a conditional belongs before the branch, and one that does not depend on the iteration belongs before the loop or per-element closure. A repeated copy costs on every pass and drifts when only one of them is edited.
- Derive a computed limit or estimate from the quantity that actually bounds it — not a request-shaped parameter that happens to be in scope, and not a counter that a second, larger counter can exceed. Where the true bound cannot be computed, document the assumption the estimate makes rather than leaving it implicit; a bound built on the wrong quantity is wrong in the direction nobody checks.

## Contribution expectations
- Preserve backward compatibility for public API (`include/svs/`)
- Prefer additive API evolution (avoid duplication of existing functionality)
- Pair public API changes with tests and documentation
- For bug fixes: add regression tests
- Run `pre-commit run --all-files` before proposing changes

## Source-of-truth files
- Build: `CMakeLists.txt`, `cmake/*.cmake` (incl. `cmake/mkl.cmake`, `cmake/multi-arch.cmake`, `cmake/numa.cmake`)
- Dependencies: `bindings/python/pyproject.toml`, `bindings/python/setup.py`
- CI: `.github/workflows/`
- Style: `.clang-format`, `.pre-commit-config.yaml`
- API: `include/svs/`, `bindings/python/`
