# Copyright (C) 2024 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit
# this software or the related documents without Intel's prior written
# permission.
#
# This software and the related documents are provided as is, with no
# express or implied warranties, other than those that are expressly stated
# in the License.

FROM quay.io/pypa/manylinux2014_x86_64:2024-07-15-c746fd8

COPY ./oneAPI.repo /etc/yum.repos.d/oneAPI.repo

# Install gcc-11
RUN yum --disablerepo=epel install -y \
    devtoolset-11-gcc \
    devtoolset-11-gcc-c++ \
    intel-oneapi-mkl \
    intel-oneapi-mkl-devel

# Enable CMake to find the config files for Intel(R) MKL.
COPY ./entry.sh .
ENTRYPOINT ["./entry.sh"]

