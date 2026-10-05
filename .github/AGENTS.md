# AGENTS.md — .github/

CI/CD workflows, automation, templates, and Copilot instructions.

- Keep required checks and matrix in workflow files only.
- Keep instruction docs long-lived (no mutable values).
- Treat workflow/job renames as breaking for tooling; change only intentionally.
- When CI behavior changes, align contributor-facing docs.
- For build-system behavior changes, suggest updating/validating CI workflows in `.github/workflows/`.
