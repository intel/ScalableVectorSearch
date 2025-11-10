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

# Source environment setup (for compiler and MKL)
source /etc/bashrc || true

# Create build+install directories for cpp runtime bindings
rm -rf /workspace/bindings/cpp/build_cpp_bindings /workspace/install_cpp_bindings
mkdir -p /workspace/bindings/cpp/build_cpp_bindings /workspace/install_cpp_bindings

# Build and install runtime bindings library
cd /workspace/bindings/cpp/build_cpp_bindings
CC=gcc CXX=g++ cmake .. \
    -DCMAKE_INSTALL_PREFIX=/workspace/install_cpp_bindings \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DBUILD_TESTING=ON
cmake --build . -j
cmake --install .

# Create tarball with symlink for compatibility
cd /workspace/install_cpp_bindings && \
ln -s lib lib64 && \
tar -czvf /workspace/svs-cpp-runtime-bindings.tar.gz .
