---
applyTo: "bindings/**"
---

# Bindings Instructions for GitHub Copilot

- A binding reports failure through its own error channel and lets nothing escape across the boundary: an entry point that cannot propagate an exception is marked `noexcept` and returns its error type. Each binding has its own channel — do not assume another's.
- A binding does not restate a default, policy, or limit the core library already selects; pass the core's own sentinel or value through, so the binding cannot drift from it.
- Give a new member of an existing family the convention its siblings already follow: how a default or "unset" is signalled, which type carries a buffer, where an output argument sits. A divergence that still compiles changes behaviour at call sites that never opted into it.
