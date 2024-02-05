FROM quay.io/pypa/manylinux2014_x86_64:2023-03-05-271004f

COPY ./oneAPI.repo /etc/yum.repos.d/oneAPI.repo

# Install gcc-11
RUN yum --disablerepo=epel install -y \
    devtoolset-11-gcc \
    devtoolset-11-gcc-c++ \
    intel-oneapi-mkl \
    intel-oneapi-mkl-devel

