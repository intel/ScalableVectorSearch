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

# FAISS validation scope for now
# Create conda env matching https://github.com/facebookresearch/faiss/blob/main/.github/actions/build_cmake/action.yml
conda create -y -n svsenv python=3.11
source /opt/conda/etc/profile.d/conda.sh
conda activate svsenv
conda config --set solver libmamba
conda install -y -c conda-forge cmake=3.30.4 make=4.2 swig=4.0 "numpy>=2.0,<3.0" scipy=1.16 pytest=7.4 gflags=2.2
conda install -y -c conda-forge gxx_linux-64=14.2 sysroot_linux-64=2.17
conda install -y mkl=2022.2.1 mkl-devel=2022.2.1

# Validate python and C++ tests against FAISS CI
git clone https://github.com/facebookresearch/faiss.git
cd faiss

echo "==============================================="
echo " Running validation of library against FAISS CI"
echo "-----------------------------------------------"
echo " FAISS Build: "
mkdir build && cd build
cmake -DBUILD_TESTING=ON -DFAISS_ENABLE_SVS=ON -DFAISS_ENABLE_GPU=OFF -DSVS_URL="file:///runtime_lib/svs-cpp-runtime-bindings${SUFFIX}.tar.gz" ..
make -j$(nproc) swigfaiss faiss_test
echo "-----------------------------------------------"
echo " FAISS C++ tests: "
./tests/faiss_test --gtest_filter=SVS.*
echo "-----------------------------------------------"
echo " FAISS-SVS C++ examples: "
make 10-SVS-Vamana-LVQ 11-SVS-Vamana-LeanVec
./tutorial/cpp/10-SVS-Vamana-LVQ
./tutorial/cpp/11-SVS-Vamana-LeanVec
echo "-----------------------------------------------"
echo " FAISS python bindings: "
cd faiss/python/
python setup.py build
echo "-----------------------------------------------"
echo " FAISS python tests: "
cd ../../../tests/
PYTHONPATH=../build/faiss/python/build/lib/ OMP_NUM_THREADS=8 python -m unittest test_svs.py
echo "-----------------------------------------------"
echo " FAISS-SVS python examples: "
cd ../tutorial/python/
PYTHONPATH=../../build/faiss/python/build/lib/ OMP_NUM_THREADS=8 python 11-SVS.py
