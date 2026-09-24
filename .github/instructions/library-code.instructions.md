---
applyTo: "include/svs/**"
---

# Library Code Instructions for GitHub Copilot

- Emit diagnostics through `svs::logging`, never to stdout or stderr. An API whose declared purpose is writing a report to a caller-supplied stream is not a diagnostic and is exempt.
- Document every parameter of a public entity, including what a default resolves to and which level of a nested structure it governs. A placeholder standing for "let the library choose" is used only where the default really is computed at call time, and says so; where the default is a fixed value, state that value.
- Do not widen a type's public surface so another layer can compute something from its internals; the layer that owns a quantity owns the operation on it. Forwarding to an owned member is fine.
- A helper, cast, or constant that more than one component needs belongs in one shared header rather than copied into each user, and a defaulted parameter is preferred over a near-duplicate overload. A type more than one component references is declared where all of them can reach it, not nested inside the first one that needed it.
- A conditional-compilation guard is a claim about what the code inside it depends on, so make the condition exactly that: not a narrower feature flag standing in for the broader platform, and not a higher instruction-set level than the surrounding path targets. Deleting such a guard removes a configuration; do not delete one as cleanup without confirming nothing builds that way.
- Name an entity for the role it plays, not the value it holds or the one algorithm that currently uses it.
- Express a requirement where the compiler enforces it — a `concept` on a template parameter, `constexpr` for a value fixed at compile time — rather than as a runtime check or a comment.
