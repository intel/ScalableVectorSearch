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

# AGENTS.md — ScalableVectorSearch

Entry point for agent context in this repo.

## Required flow
1. Read this file.
2. Read the nearest `AGENTS.md` for touched files.
3. Keep diffs scoped and minimal.
4. Validate with smallest relevant build/test target.
5. If behavior/API changes, update tests/docs in the same PR.

## Mutable details: source of truth only
- Build/options: `CMakeLists.txt`, `cmake/*.cmake`
- CI/required checks: `.github/workflows/`
- Style/lint: `.clang-format`, `.pre-commit-config.yaml`
- Public C++ API: `include/svs/`
- Python package/runtime: `bindings/python/`

Do not duplicate versions, matrices, or flags in instructions.

## Directory map
- `.github/AGENTS.md`
- `benchmark/AGENTS.md`
- `bindings/AGENTS.md`
- `cmake/AGENTS.md`
- `examples/AGENTS.md`
- `include/svs/AGENTS.md`
- `tests/AGENTS.md`
- `tools/AGENTS.md`
- `utils/AGENTS.md`
