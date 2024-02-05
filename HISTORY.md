# SVS 0.0.2 Release Notes

## `pysvs` (Python)

* Deprecated `num_threads` keyword argument from `pysvs.VamanaBuildParameters` and added
  `num_threads` keyword to `pysvs.Vamana.build`.
* Exposed the `prune_to` parameter for `pysvs.VamanaBuildParameters` (see description below
  for an explanation of this change).
* Added preliminary support for building `pysvs.Flat` and `pysvs.Vamana` directly from
  `np.float16` arrays.

## `libsvs` (C++)

### Breaking Changes

* Removed `nthreads` member of `VamanaBuildParameters` and added the number of threads as
  an argument to `svs::Vamana::build`/`svs::Vamana::build`.
* Added a `prune_to` argument to `VamanaBuildParameters`. This can be set to a value less
  than graph_max_degree (heuristically, setting this to be 4 less is a good trade-off
  between accuracy and speed). When pruning is performed, this parameter is used to
  determine the number of candidates to generate after pruning. Setting this less than
  `graph_max_degree` greatly reduces the time spent when managing backedges.
* Improved pruning rules for Euclidean and InnerProduct. Vamana index construction should
  be faster and yield slightly improved indexes.
* Added an experimental external-threading interface to `svs::index::VamanaIndex`.
* Overhauled extension mechanisms using a `tag_invoke` style approach. This decouples the
  `svs::index::VamanaIndex` implementation from extensions like LVQ, reducing header
  dependence and improving precision of algorithm customization.

### Save/Load API
* Enabled context-free saving and loading of simple data structures. This allows simple
  data structures to be saved and reloaded from TOML files without requiring access to the
  saving/loading directory. Classes implementing this saving and loading allow for more
  flexible storage.
* Overhauled the implementation of saving and loading to enable more scalable implementation.
* `svs::data::SimpleData` family of data structures are now directly saveable and loadable
  and no longer require proxy-classes.

**Breaking Serialization Changes**

* Changed LVQ-style datasets from `v0.0.1` to `v0.0.2`: Removed centroids from being stored
  with the ScaledBiasedCompressedDataset.  Centroids are now stored in the higher level LVQ
  dataset.

### Back-end Changes

Changes to library internals that do not necessarily affect the top level API but could
affect performance or users relying on internal APIs.

* Improved the performance of the LVQ inner-product implementation.
* Moved dynamic uispatcher from the Python bindings into `libsvs`.
* Data structure loading has been augmented with the `svs::lib::Lazy` class, allowing for
  arbitrary deferred work to be executed when loading data structures.
* Removed the old "access mode" style API for multi-level datasets, instead using
  `tag_invoke` for customization.
* Reduced binary footprint by removing `std::function` use for general multi-threaded
  functions.
* Updated `ANNException` to use `fmtlib` style message directly rather than `std::ostream`
  style overloading. The new syntax turns
  ```c++
  ANNEXCEPTION("Expected ", a, ", got ", b, "!");
  ```
  to
  ```c++
  ANNEXCEPTION("Expected {}, got {}!", a, b);
  ```

## Binaries and Utilities

* Added a benchmarking framework in `/benchmark` to automatically run and aggregate index
  construction and search for large scale benchmarks. Documentation is currently sparse
  but planned.

## Third Party

* Bump [fmtlib](https://github.com/fmtlib/fmt) from 9.1.0 to 10.1.1.

