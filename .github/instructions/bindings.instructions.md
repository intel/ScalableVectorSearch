---
applyTo: "bindings/**"
---

# Bindings Instructions for GitHub Copilot

- A binding reports failure through its own error channel and lets nothing escape across the boundary: an entry point that cannot propagate an exception is marked `noexcept` and returns its error type. Each binding has its own channel — do not assume another's.
- A binding does not restate a default, policy, or limit the core library already selects; pass the core's own sentinel or value through, so the binding cannot drift from it.
