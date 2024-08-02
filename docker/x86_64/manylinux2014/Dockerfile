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

