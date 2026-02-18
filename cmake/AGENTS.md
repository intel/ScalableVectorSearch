# AGENTS.md — cmake/

Build modules, dependency wiring, and feature toggles.

- `CMakeLists.txt` + `cmake/*.cmake` are authoritative.
- Keep option names/defaults stable unless task requires change.
- Prefer additive options over rewrites.
- Validate option/target changes against CI workflows.
