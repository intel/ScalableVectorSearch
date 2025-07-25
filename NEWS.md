# SVS 0.0.8 Release Notes

## Additions and Changes

* Addition of scalar quantization dataset with int8 support

* Introduced multi-vector index and batch iterator support

* Automatic ISA dispatching with optimizations based on AVX support

* Enhanced logging capabilities

* Updated vamana iterator API

* Broader [shared library](https://github.com/intel/ScalableVectorSearch/releases) support:

  * gcc-11+, clang-18+, glibc 2.26+ compatibility
  
  * Static library provided in addition to .so
  
  * Intel(R) MKL linked within the shared library - no need for Intel(R) MKL in user environment
