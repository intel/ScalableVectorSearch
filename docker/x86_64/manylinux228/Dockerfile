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

# Use --build-arg REGISTRY="quay.io/" to revert to default
ARG REGISTRY=cache-registry.caas.intel.com/cache-quay/
FROM ${REGISTRY}pypa/manylinux_2_28_x86_64

ENV http_proxy=http://proxy-dmz.intel.com:911
ENV https_proxy=http://proxy-dmz.intel.com:912
ENV no_proxy=localhost,127.0.0.1

# Install CMake3
RUN yum install -y cmake3 \
    && (if [ -x /usr/local/bin/cmake ]; then rm /usr/local/bin/cmake; fi) \
    && ln -sf /usr/bin/cmake3 /usr/local/bin/cmake

# Install gcc-11
RUN yum install -y gcc-toolset-11 valgrind wget

# Install Intel(R) MKL
COPY ./ScalableVectorSearch/docker/x86_64/manylinux2014/oneAPI.repo /etc/yum.repos.d/oneAPI.repo
RUN yum install -y intel-oneapi-mkl-core-devel-2024.1

# Configure environment setup properly without recursion
RUN echo '# Configure gcc-11' > /etc/profile.d/01-gcc.sh && \
    echo 'source scl_source enable gcc-toolset-11' >> /etc/profile.d/01-gcc.sh && \
    chmod +x /etc/profile.d/01-gcc.sh

# Copy sde script to run in test script later
# Need the 32-bit compatibility libraries to make sde work
RUN yum install -y glibc.i686
COPY ./.github/scripts/run_sde.sh /tmp/run_sde.sh
