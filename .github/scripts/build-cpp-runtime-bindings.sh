#!/bin/bash
# Copyright 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e  # Exit on error

# Source environment setup (for compiler)
source /etc/bashrc || true

# Source MKL environment (required for IVF)
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh --include-intel-llvm 2>/dev/null || true
    echo "MKL sourced: MKLROOT=${MKLROOT}"
else
    echo "ERROR: MKL setvars.sh not found"
    exit 1
fi

# Create build+install directories for cpp runtime bindings
rm -rf /workspace/bindings/cpp/build_cpp_bindings /workspace/install_cpp_bindings
mkdir -p /workspace/bindings/cpp/build_cpp_bindings /workspace/install_cpp_bindings

# Build and install runtime bindings library
cd /workspace/bindings/cpp/build_cpp_bindings

# Set default cmake args and add SVS_URL if specified
CMAKE_ARGS=(
    "-DSVS_BUILD_RUNTIME_TESTS=ON"
    "-DCMAKE_INSTALL_PREFIX=/workspace/install_cpp_bindings"
    "-DCMAKE_INSTALL_LIBDIR=lib"
    "-DSVS_RUNTIME_ENABLE_LVQ_LEANVEC=${ENABLE_LVQ_LEANVEC:-ON}"
    "-DSVS_RUNTIME_ENABLE_IVF=ON"
)

if [ -n "$SVS_URL" ]; then
    CMAKE_ARGS+=("-DSVS_URL=$SVS_URL")
fi

# Build and install runtime bindings library (from bindings/cpp)
if [ -n "$CC" ]; then
    echo "Using CC=${CC} and CXX=${CXX} for building cpp runtime bindings"
else
    echo "Using default compiler for building cpp runtime bindings"
fi
CC=${CC:-gcc} CXX=${CXX:-g++} cmake .. "${CMAKE_ARGS[@]}"
cmake --build . -j
cmake --install .

# Build conda package for cpp runtime bindings
source /opt/conda/etc/profile.d/conda.sh
cd /workspace
# Free disk space by removing large build artifacts not needed for ctest or conda build.
# Keep the test binary (tests/) so the workflow can run ctest in a subsequent step.
find /workspace/bindings/cpp/build_cpp_bindings -name '*.o' -delete 2>/dev/null || true
find /workspace/bindings/cpp/build_cpp_bindings -name '*.a' -delete 2>/dev/null || true
find /workspace/bindings/cpp/build_cpp_bindings -name '*.so*' -not -path '*/tests/*' -not -name 'libsvs_runtime*' -delete 2>/dev/null || true
# Use /workspace for temp files to avoid filling up /tmp during LTO compilation
mkdir -p /workspace/tmp
TMPDIR=/workspace/tmp ENABLE_LVQ_LEANVEC=${ENABLE_LVQ_LEANVEC:-ON} SVS_URL="${SVS_URL}" SUFFIX="${SUFFIX}" conda build bindings/cpp/conda-recipe --output-folder /workspace/conda-bld

# Create tarball with symlink for compatibility
cd /workspace/install_cpp_bindings && \
ln -s lib lib64 && \
tar -czvf /workspace/svs-cpp-runtime-bindings${SUFFIX}.tar.gz .
