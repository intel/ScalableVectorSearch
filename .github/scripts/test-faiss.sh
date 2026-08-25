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

# Prefix can be set to validate on different architectures via SDE
RUN_PREFIX="${RUN_PREFIX:-}"

# Source environment setup (for compiler and MKL)
source /etc/bashrc || true

# FAISS validation scope for now
# Create conda env matching https://github.com/facebookresearch/faiss/blob/main/.github/actions/build_cmake/action.yml
conda create -y -n svsenv python=3.11
source /opt/conda/etc/profile.d/conda.sh
conda activate svsenv
conda config --set solver libmamba
conda install -y -c conda-forge cmake=3.30.4 make=4.2 swig=4.0 "numpy>=2.0,<3.0" scipy=1.16 pytest=7.4 gflags=2.2 setuptools
conda install -y -c conda-forge gxx_linux-64=14.2 sysroot_linux-64=2.17
conda install -y mkl=2025.3 mkl-devel=2025.3

# Install libsvs-runtime from local conda package
conda install -y /runtime_conda/libsvs-runtime-*.conda

# Validate python and C++ tests against FAISS CI
git clone https://github.com/facebookresearch/faiss.git
cd faiss

echo "==============================================="
echo " Running validation of library against FAISS CI"
echo "-----------------------------------------------"
echo " FAISS Build: "
mkdir build && cd build
cmake -DBUILD_TESTING=ON -DFAISS_ENABLE_SVS=ON -DFAISS_ENABLE_GPU=OFF ..
make -j$(nproc) swigfaiss faiss_test
echo "-----------------------------------------------"
echo " FAISS C++ tests: "
./tests/faiss_test --gtest_filter=SVS.*
echo "-----------------------------------------------"
echo " FAISS-SVS C++ examples: "
make 10-SVS-Vamana-LVQ 11-SVS-Vamana-LeanVec
# The examples request LVQ/LeanVec unconditionally, which needs an enabled runtime and
# Intel hardware; public-only compiles the formats out, so the CPU vendor alone mispredicts.
lvq_leanvec_missing=""
if [ "${ENABLE_LVQ_LEANVEC:-ON}" != "ON" ]; then
  lvq_leanvec_missing="the runtime is built without LVQ/LeanVec support"
elif ! grep -q "GenuineIntel" /proc/cpuinfo; then
  lvq_leanvec_missing="the CPU is not GenuineIntel"
fi

# The trailing class name differs between bindings/cpp/src/vamana_index.cpp and
# dynamic_vamana_index.cpp, so matching the full message would pin the examples to one index kind.
storage_kind_rejection="The specified storage kind is not compatible with the"

# A nonzero exit alone would let a broken example, a missing library or a segfault pass as expected.
expect_storage_kind_rejection() {
  local label="$1"
  shift
  local output status
  output=$("$@" 2>&1) && status=0 || status=$?
  echo "$output"
  if [ "$status" -eq 0 ]; then
    echo "UNEXPECTED: $label succeeded although $lvq_leanvec_missing"
    return 1
  fi
  if ! printf '%s\n' "$output" | grep -qF "$storage_kind_rejection"; then
    echo "UNEXPECTED: $label exited $status without rejecting the storage kind"
    return 1
  fi
  echo "XFAIL: $label rejected the storage kind as expected ($lvq_leanvec_missing)"
}

if [ -z "$lvq_leanvec_missing" ]; then
  $RUN_PREFIX ./tutorial/cpp/10-SVS-Vamana-LVQ
  $RUN_PREFIX ./tutorial/cpp/11-SVS-Vamana-LeanVec
else
  echo "LVQ/LeanVec examples expected to reject the storage kind: $lvq_leanvec_missing"
  expect_storage_kind_rejection 10-SVS-Vamana-LVQ \
    $RUN_PREFIX ./tutorial/cpp/10-SVS-Vamana-LVQ
  expect_storage_kind_rejection 11-SVS-Vamana-LeanVec \
    $RUN_PREFIX ./tutorial/cpp/11-SVS-Vamana-LeanVec
fi
echo "-----------------------------------------------"
echo " FAISS python bindings: "
cd faiss/python/
python setup.py build
echo "-----------------------------------------------"
echo " FAISS python tests: "
cd ../../../tests/
PYTHONPATH=../build/faiss/python/build/lib/ OMP_NUM_THREADS=4 python -m unittest test_svs_py.py
echo "-----------------------------------------------"
echo " FAISS-SVS python examples: "
cd ../tutorial/python/
if [ -z "$lvq_leanvec_missing" ]; then
  PYTHONPATH=../../build/faiss/python/build/lib/ OMP_NUM_THREADS=4 $RUN_PREFIX python 11-SVS.py
else
  echo "SVS python example expected to reject the storage kind: $lvq_leanvec_missing"
  expect_storage_kind_rejection 11-SVS.py \
    env PYTHONPATH=../../build/faiss/python/build/lib/ OMP_NUM_THREADS=4 \
    $RUN_PREFIX python 11-SVS.py
fi
