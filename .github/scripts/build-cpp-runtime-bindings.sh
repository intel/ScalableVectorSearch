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

source /opt/conda/etc/profile.d/conda.sh

rm -rf /workspace/conda-bld /workspace/install_cpp_bindings /workspace/build_cpp_bindings_tests
mkdir -p /workspace/conda-bld /workspace/install_cpp_bindings /workspace/build_cpp_bindings_tests

# Single LTO build of libsvs_runtime via the conda recipe. The resulting .conda
# is the canonical artifact; the standalone tarball below is a re-pack of its
# install tree, so the two distribution methods ship byte-identical libraries.
cd /workspace
mkdir -p /workspace/tmp
TMPDIR=/workspace/tmp \
ENABLE_LVQ_LEANVEC="${ENABLE_LVQ_LEANVEC:-ON}" \
SVS_URL="${SVS_URL}" \
SUFFIX="${SUFFIX}" \
    conda build bindings/cpp/conda-recipe --output-folder /workspace/conda-bld

# Extract the conda payload into a plain install prefix; conda-package-handling
# is part of the base conda env.
CONDA_PKG=$(ls /workspace/conda-bld/linux-64/libsvs-runtime-*.conda | head -n 1)
if [ -z "${CONDA_PKG}" ]; then
    echo "ERROR: conda-build did not produce a libsvs-runtime package"
    exit 1
fi
/opt/conda/bin/python -m conda_package_handling.api extract "${CONDA_PKG}" --dest /workspace/install_cpp_bindings
# Drop conda-only metadata so the tarball matches a plain `cmake --install` tree.
rm -rf /workspace/install_cpp_bindings/info

# Tests are built separately against the installed prefix so they don't trigger a
# second LTO link of libsvs_runtime.
cd /workspace/build_cpp_bindings_tests
CC=gcc CXX=g++ cmake /workspace/bindings/cpp/tests \
    -Dsvs_runtime_DIR=/workspace/install_cpp_bindings/lib/cmake/svs_runtime \
    -DSVS_RUNTIME_ENABLE_IVF=ON
cmake --build . -j

# Tarball the install tree (the recipe already includes the lib64 -> lib symlink).
cd /workspace/install_cpp_bindings
tar -czvf /workspace/svs-cpp-runtime-bindings${SUFFIX}.tar.gz .
