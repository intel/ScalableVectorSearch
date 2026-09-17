---
applyTo: "{tests/**,bindings/*/tests/**}"
---

# Testing Instructions for GitHub Copilot

- Add regression tests for bug fixes.
- Keep tests deterministic.
- Prefer the smallest test surface that proves correctness for a given variant, then cover every variant the subject documents — each index type, each dataset or quantization kind the shared reference helpers accept — rather than the default alone.
- Drive the subject into the state under test rather than only calling the API that would reach it: enough input to cross a batch boundary for per-batch behaviour, a mutation before the save/load round trip for a structure that supports mutation.
- A new test goes through the existing shared fixture or parametrized helper for its subject rather than reimplementing the setup. Where no helper covers the subject, or where reusing one would defeat what the test asserts, bespoke setup is correct.
- Refer to repository test/build configuration as source of truth rather than duplicating values.
