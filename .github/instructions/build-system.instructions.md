---
applyTo: "{CMakeLists.txt,cmake/**,.github/workflows/**}"
---

# Build System Instructions for GitHub Copilot

- Use existing option/target patterns; avoid introducing parallel build paths. Before adding a compile flag, option, or CI setup step, check whether an existing one already covers the condition; where the same block genuinely exists in more than one file, either change every copy or factor it into one included file.
- Keep configuration values referenced from existing CMake modules.
- Return a computed value another module consumes from a `function()` via `PARENT_SCOPE`, or publish it as a cache entry; a bare `set()` at file scope is not an interface, and a later module setting the same name collides with it silently.
- Do not hardcode versions/toolchain assumptions in instructions or comments.
- For behavior changes, suggest updating/validating CI workflows in `.github/workflows/`.
