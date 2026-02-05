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

These examples utilize LVQ and LeanVec interfaces which are available when linking to a SVS shared/static library, which are published with [releases](https://github.com/intel/ScalableVectorSearch/releases), as a tarball, pip wheel, or conda package. Note that these examples will _not_ run after building the open source codebase without the shared/static library. These examples include:

### Vamana Examples
- [example_vamana_with_compression.cpp](./example_vamana_with_compression.cpp): Demonstrates building, searching, saving, and reloading a Vamana index with a LeanVec-compressed dataset.
- [example_vamana_with_compression_lvq.cpp](./example_vamana_with_compression_lvq.cpp): Demonstrates building, searching, saving, and reloading a Vamana index with a LVQ-compressed dataset.
- [example_vamana_with_compression_dynamic.cpp](./example_vamana_with_compression_dynamic.cpp): Demonstrates building, searching, saving, and reloading a dynamic Vamana index (allows vector insertions and deletions over time) with a LeanVec-compressed dataset.

### IVF Examples (require SVS_EXPERIMENTAL_ENABLE_IVF=ON)
- [example_ivf_with_compression_lvq.cpp](./example_ivf_with_compression_lvq.cpp): Demonstrates building, searching, saving, and reloading a Dynamic IVF index with a LVQ-compressed dataset.
- [example_ivf_with_compression_leanvec.cpp](./example_ivf_with_compression_leanvec.cpp): Demonstrates building, searching, saving, and reloading a Dynamic IVF index with a LeanVec-compressed dataset.
- [example_ivf_with_compression_dynamic.cpp](./example_ivf_with_compression_dynamic.cpp): Demonstrates building, searching, adding/deleting vectors, compacting, saving, and reloading a Dynamic IVF index with a LeanVec-compressed dataset (similar to the dynamic Vamana example).

See [CMakeLists.txt](./CMakeLists.txt) for details on linking to the SVS shared library.

## Running the examples

The CMakeLists.txt is set up to detail usage of all options (tarball, pip, conda), and will prioritize utilization in the following order:

1. **System/Conda/Pip installation** - If SVS is installed in a standard location that CMake can find
2. **GitHub Release download** - If not found, it will  download the tarball from GitHub

### Option 1: Using libsvs in a conda environment

Install the `libsvs` package:
```bash
conda install -c https://software.repos.intel.com/python/conda libsvs

mkdir build
cd build
CC=gcc-11 CXX=g++-11 cmake ../
make -j
./example_vamana_with_compression_dynamic
```

### Option 2: Using pip-installed libsvs

Install the `libsvs` package and ensure CMake can find it by setting `CMAKE_PREFIX_PATH`:
```bash
pip install --index-url https://software.repos.intel.com/python/pypi libsvs

mkdir build
cd build
# Note that pip packages require setting CMAKE_PREFIX_PATH to find the library, conda handles this automatically
CC=gcc-11 CXX=g++-11 cmake -DCMAKE_PREFIX_PATH=$(python -c "import libsvs; print(libsvs.get_cmake_prefix_path())") ..
make -j
./example_vamana_with_compression_dynamic
```

### Option 3: Using shared library tarball

If `libsvs` is not installed, CMake will download the tarball (see [CMakeLists.txt](./CMakeLists.txt) for the necessary steps here):
```bash
mkdir build
cd build
CC=gcc-11 CXX=g++-11 cmake ../
make -j
./example_vamana_with_compression_dynamic
```
