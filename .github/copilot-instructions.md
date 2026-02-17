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

# GitHub Copilot Instructions — ScalableVectorSearch

## Mandatory
1. Read root `AGENTS.md` first.
2. Read the nearest directory `AGENTS.md` for edited files.
3. If multiple apply, use the most specific.

## Authoring rules
- Keep suggestions minimal and scoped.
- Use source-of-truth files for mutable details.
- Do not invent or hardcode versions/flags/matrices.

## Source-of-truth files
- `CMakeLists.txt`, `cmake/*.cmake`
- `.github/workflows/`
- `.clang-format`, `.pre-commit-config.yaml`
- `include/svs/`, `bindings/python/`
