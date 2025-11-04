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
# Create conda env matching https://github.com/facebookresearch/faiss/blob/main/.github/actions/build_cmake/action.yml
conda create -y -n svsenv python=3.11
source /opt/conda/etc/profile.d/conda.sh
conda activate svsenv
conda config --set solver libmamba
conda install -y -c conda-forge cmake=3.30.4 make=4.2 swig=4.0 "numpy>=2.0,<3.0" scipy=1.16 pytest=7.4 gflags=2.2
conda install -y -c conda-forge gxx_linux-64=14.2 sysroot_linux-64=2.17
conda install -y mkl=2022.2.1 mkl-devel=2022.2.1

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
