<!--
  ~ Copyright 2025 Intel Corporation
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

These examples utilize LVQ and LeanVec interfaces which are available when linking to a SVS shared/static library, which are published with [releases](https://github.com/intel/ScalableVectorSearch/releases). Note that these examples will _not_ run after building the open source codebase without the shared/static library. These examples include:
- [example_vamana_with_compression.cpp](./example_vamana_with_compression.cpp): Demonstrates building, searching, saving, and reloading an index with a LeanVec-compressed dataset.
- [example_vamana_with_compression_lvq.cpp](./example_vamana_with_compression_lvq.cpp): Demonstrates building, searching, saving, and reloading an index with a LVQ-compressed dataset.
- [example_vamana_with_compression_dynamic.cpp](./example_vamana_with_compression_dynamic.cpp): Demonstrates building, searching, saving, and reloading a dynamic index (allows vector insertions and deletions over time) with a LeanVec-compressed dataset.

See [CMakeLists.txt](./CMakeLists.txt) for details on linking to the SVS shared library and follow the commands below to compile and use the SVS shared library to run shared.cpp example:

```
mkdir build
cd build
CC=gcc-11 CXX=g++-11 cmake ../
make -j
./shared
```
