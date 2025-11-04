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

set -e

# Source environment setup (for compiler and MKL)
source /etc/bashrc || true

# chmod +x docker/x86_64/list-dependencies.sh
# ./docker/x86_64/list-dependencies.sh

# FAISS validation scope for now
# TODO: point to root repo eventually 
git clone -b rfsaliev/svs-faiss-bindings https://github.com/ahuber21/faiss.git
cd faiss
sed -i "s|set(SVS_URL .*|set(SVS_URL \"file:///runtime_lib/svs-cpp-runtime-bindings${PLATFORM_NAME}.tar.gz\" CACHE STRING \"SVS URL\")|" faiss/CMakeLists.txt

echo "================================================"
echo " Runnning validation of library against FAISS CI"
echo "------------------------------------------------"
echo " FAISS Build: "
mkdir build && cd build
# TODO: create conda env
cmake -DBUILD_TESTING=ON -DFAISS_ENABLE_SVS=ON -DFAISS_ENABLE_GPU=OFF ..
make -j swigfaiss
echo "------------------------------------------------"
echo " FAISS python bindings: "
cd faiss/python/
python setup.py build
echo "------------------------------------------------"
echo " FAISS python tests: "
cd ../../../tests/
PYTHONPATH=../build/faiss/python/build/lib/ OMP_NUM_THREADS=8 python -m unittest test_svs.py

# TODO: C++ tests
