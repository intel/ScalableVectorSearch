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
# Check if running on Intel hardware (LVQ/LeanVec require Intel-specific instructions)
if grep -q "GenuineIntel" /proc/cpuinfo; then
  ./tutorial/cpp/10-SVS-Vamana-LVQ
  ./tutorial/cpp/11-SVS-Vamana-LeanVec
else
  echo "Non-Intel CPU detected - LVQ/LeanVec examples expected to fail"
  set +e
  ./tutorial/cpp/10-SVS-Vamana-LVQ
  exit_code_10=$?
  ./tutorial/cpp/11-SVS-Vamana-LeanVec
  exit_code_11=$?
  set -e
  
  if [ $exit_code_10 -ne 0 ] && [ $exit_code_11 -ne 0 ]; then
    echo "XFAIL: Examples failed as expected on non-Intel hardware"
  else
    echo "UNEXPECTED: One or more tests passed on non-Intel hardware (exit codes: $exit_code_10, $exit_code_11)"
    exit 1
  fi
fi
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
if grep -q "GenuineIntel" /proc/cpuinfo; then
  PYTHONPATH=../../build/faiss/python/build/lib/ OMP_NUM_THREADS=8 python 11-SVS.py
else
  echo "Non-Intel CPU detected - SVS python example expected to fail"
  set +e
  PYTHONPATH=../../build/faiss/python/build/lib/ OMP_NUM_THREADS=8 python 11-SVS.py
  exit_code=$?
  set -e
  
  if [ $exit_code -ne 0 ]; then
    echo "XFAIL: Python example failed as expected on non-Intel hardware"
  else
    echo "UNEXPECTED: Python example passed on non-Intel hardware"
    exit 1
  fi
fi
