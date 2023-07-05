# SVS 0.0.2 Release Notes

## C++ Library

* Enabled serialization of `VamanaBuildParameters`.
* Removed `nthreads` member of `VamanaBuildParameters` and added the number of threads as
  an argument to `svs::index::Vamana::build`.

## PYSVS

* Deprecated `num_threads` keyword argument from `pysvs.VamanaBuildParameters` and added
  `num_threads` keyword to `pysvs.Vamana.build`.


