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

# AGENTS.md — bindings/

Interop layer between core C++ and exposed APIs.

- Read: `bindings/cpp/AGENTS.md`, `bindings/python/AGENTS.md`.
- Keep contracts aligned with `include/svs/`.
- Make ownership/lifetime explicit across boundaries.
- Pair behavior/signature changes with tests (and examples when user-facing).
