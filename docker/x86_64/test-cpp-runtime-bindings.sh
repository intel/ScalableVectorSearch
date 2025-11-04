#!/bin/bash
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
