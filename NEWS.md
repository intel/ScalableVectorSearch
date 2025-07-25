# SVS 0.0.8 Release Notes

## Additions and Changes

* Addition of 8-bit scalar quantization support to C++ interface

* Introduced multi-vector index and batch iterator support

* Automatic ISA dispatching with optimizations based on AVX support

* Enabled compatibility with ARM and MacOS

* Enhanced logging capabilities

* Updated vamana iterator API

* Broader [shared library](https://github.com/intel/ScalableVectorSearch/releases) support:

  * gcc-11+, clang-18+, glibc 2.26+ compatibility
  
  * Static library provided in addition to .so
  
  * Intel(R) MKL linked within the shared library - no need for Intel(R) MKL in user environment
