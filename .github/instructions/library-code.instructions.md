---
applyTo: "include/svs/multi-arch/**"
---

# Multi-Arch Instructions for GitHub Copilot

- Include the header that defines any macro which expands into a translation unit's own declarations or definitions. This covers code-generating macros, not inline annotations such as a force-inline, unused, or forwarding marker. A generating macro whose defining header is absent hides what the file actually defines, so its contents cannot be determined by reading the file alone.
