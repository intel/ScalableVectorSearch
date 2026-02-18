---
applyTo: "{CMakeLists.txt,cmake/**}"
---

# Build System Instructions for GitHub Copilot

- Use existing option/target patterns; avoid introducing parallel build paths.
- Keep configuration values referenced from existing CMake modules.
- Do not hardcode versions/toolchain assumptions in instructions or comments.
- For behavior changes, suggest updating/validating CI workflows in `.github/workflows/`.
