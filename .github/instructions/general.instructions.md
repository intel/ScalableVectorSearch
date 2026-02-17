<!--
  ~ Copyright 2026 Intel Corporation
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

---
applyTo: "**"
---

# General Instructions for GitHub Copilot

## Context loading (required)
- Read root `AGENTS.md` first.
- Then read nearest local `AGENTS.md` for touched files.
- If multiple apply, follow the most specific one.

## Long-lived policy
- Keep this file stable and high-level.
- Put mutable details only in source-of-truth files.
- Reference those files instead of duplicating values.

## Source-of-truth
- Build/options: `CMakeLists.txt`, `cmake/*.cmake`
- CI/checks: `.github/workflows/`
- Style/lint: `.clang-format`, `.pre-commit-config.yaml`
- API/package contracts: `include/svs/`, `bindings/python/`

## Change hygiene
- Prefer minimal scoped edits.
- Preserve backward compatibility by default.
- Pair behavior/API changes with tests/docs updates.
