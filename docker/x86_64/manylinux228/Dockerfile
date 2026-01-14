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

FROM quay.io/pypa/manylinux_2_28_x86_64:2025.12.29-1

# Install CMake3
RUN yum install -y cmake-3.26.5 \
    && yum clean all \
    && (if [ -x /usr/local/bin/cmake ]; then rm /usr/local/bin/cmake; fi) \
    && ln -sf /usr/bin/cmake3 /usr/local/bin/cmake

# Install gcc-11
RUN yum install -y gcc-toolset-11-11.1-1.el8 wget-1.19.5 \
    && yum clean all

# Configure environment setup properly without recursion
RUN echo '# Configure gcc-11' > /etc/profile.d/01-gcc.sh && \
    echo 'source scl_source enable gcc-toolset-11' >> /etc/profile.d/01-gcc.sh && \
    chmod +x /etc/profile.d/01-gcc.sh

# Download and install Miniforge
RUN wget -nv https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh && \
    /opt/conda/bin/conda install -y conda-build && \
    /opt/conda/bin/conda clean -afy
ENV PATH="/opt/conda/bin:$PATH"
