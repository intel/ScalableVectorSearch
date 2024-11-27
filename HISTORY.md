# SVS 0.0.4 Release Notes

Note that `pysvs` was changed to `svs` since this release.

## Major Changes

### Serialization Update

The serialization strategy for all SVS serialized objects has been updated from `v0.0.1` to
`v0.0.2`. In order to be compatible, all previously saved objects will need to be updated.

Updating can be done using the `pysvs` upgrade tool:
```python
import pysvs
pysvs.upgrader.upgrade("path-to-save-directory")
```
Some notes about the upgrade process are given below.

* If the object does not need upgrading, no changes will be made.
* If the object *does* need upgrading, then only the TOML file will be modified. The upgrade
  tool fill first create a backup (by changing the extension to `.backup.toml`) and
  then attempt an upgrade.

  If the upgrade fails, please let the maintainers know so we can fix the upgrade tool.

  Furthermore, if a backup file already exists, the upgrade process will abort.
* Objects upgraded to the `v0.0.2` serialization format are *forwards* compatible.
  This means that they can still be loaded by *older* version of `pysvs`.

**Why is this change needed?**

This change is necessary to support efficient introspective loading, where serialized
objects can be inspected for load compatibility. This, in turn, enables automatic loading
of previously serialized SVS objects.

### Build System and Testing

Included reference results for the Vamana index require Intel(R) MKL 2024.1 for reproducibility in testing.
Linking against Intel(R) MKL 2023.X may cause LeanVec tests to fail.

Reference results for your version of Intel(R) MKL can be regenerated using
```sh
# Build the test generators
mkdir build
cd build
CC=gcc-11 CXX=g++-11 cmake .. -DCMAKE_BUILD_TYPE=Release -DSVS_BUILD_BENCHMARK_TEST_GENERATORS=YES -DSVS_EXPERIMENTAL_LEANVEC=YES
make -j

# Run the test generator executable.
./benchmark/svs_benchmark vamana_test_generator ../tools/benchmark_inputs/vamana/test-generator.toml vamana_reference.toml 5 ../data/test_dataset
cp ./vamana_reference ../data/test_dataset/reference/vamana_reference.toml
```

### Logging Infrastructure

SVS has switched to using [spdlog](https://github.com/gabime/spdlog) for its logging needs.
As such, users will now have control of what messages get logged and where they are logged.
The default interface for configuring logging is through the environment variables `SVS_LOG_LEVEL` and `SVS_LOG_SINK`.
Valid values for `SVS_LOG_LEVEL` in order of increasing severity are shown below:


| `SVS_LOG_LEVEL`       |  Descriptions                                                 |
| --------------------- | ------------------------------------------------------------- |
| ``TRACE``             | Tracing control flow through functions. Verbose.              |
| ``DEBUG``             | Verbose logging useful for debugging. Verbose.                |
| ``INFO``              | Informative prints for long-running processed.                |
| ``WARN`` (default)    | Diagnostic prints that may need to be addressed by the user.  |
| ``ERROR``             | Program errors.                                               |
| ``CRITICAL``          | Critical information.                                         |
| ``OFF``               | Disable logging.                                              |

Logging sinks control where logged message get sent and can be controlled using `SVS_LOG_SINK` with the following values.

| `SVS_LOG_SINK`            | Description
| ------------------------- | ----------------------------------------- |
| ``stdout`` (default)      | Send all messages to ``stdout``           |
| ``stderr``                | Send all messages to ``stderr``           |
| ``null``                  | Suppress all logging messages.            |
| ``file:/path/to/file``    | Send all messages to `/path/to/file`.     |

Additionally, both the C++ library and `pysvs` contain APIs for customizing logging that supersede the environment variables.
In C++, any `std::shared_ptr<spdlog::logger>` can be used if desired.

Finally, if environment variable based initialization is not desired, it can be disabled by providing `-DSVS_INITIALIZE_LOGGER=NO` to CMake at configuration time.

### Performance Enhancements

* Generally improved the performance of uncompressed distance computations with run-time lengths for Intel(R) AVX-512 based systems.
* Fixed a performance pathology for run-time dimensional sequential LVQ4(xN) when the number of dimensions is not a multiple of 16.

## `pysvs` (Python)

### Additions and Changes

* Reloading a previously saved index no longer requires exact reconstruction of the original
  loader.

  Previously, if an index was constructed and saved using the following
  ```python
  # Load data using online compression.
  loader = pysvs.VectorDataLoader(...)
  lvq = pysvs.LVQLoader(loader, primary = 4, residual = 8)

  # Build the index.
  parameters = pysvs.VamanaBuildParameters(...)
  index = pysvs.Vamana.build(params, lvq, pysvs.L2, num_threads = 10)

  # Save the result to three directories.
  index.save("config", "graph", "data")
  ```
  Then the index must be reloaded using
  ```python
  lvq = pysvs.LVQLoader("data", primary = 4, residual = 8)
  index = pysvs.Vamana("config", pysvs.GraphLoader("graph"), lvq, distance = pysvs.L2)
  ```
  Now, the following will work
  ```python
  index = pysvs.Vamana(
      "config",
      "graph",   # No longer need explicit `pysvs.GraphLoader`
      "data",    # SVS discovers this is LVQ data automatically
      distance = pysvs.L2
  )
  ```
  To tailor the run-time parameters of reloaded data (for example, the strategy and padding
  used by LVQ), automatic inference of identifying parameters makes this easier:
  ```python
  lvq = pysvs.LVQLoader(
      "data",   # Parameters `primary`, `residual`, and `dims` are discovered automatically
      strategy = pysvs.LVQStrategy.Sequential,
      padding = 0
  )
  index = pysvs.Vamana("config", "graph", loader)
  ```

* The Vamana and DynamicVamana indexes now have a reconstruction interface.
  This has the form
  ```python
  index = pysvs.Vamana(...)
  vectors = index.reconstruct(I)
  ```
  where `I` is a arbitrary dimenaional `numpy` array of `uint64` indices.
  This API returns reconstructed vectors as a `numpy` array with the shape
  ```python
  vectors.shape == (*I.shape, index.dimensions())
  ```

  In particular, the following now works:
  ```python
  I, D = index.search(...)
  vectors = index.reconstruct(I)
  ```
  **Requirements**
  * For `pysvs.Vamana` the indices in `I` must all in `[0, index.size())`.
  * For `pysvs.DynamicVamana`, the in `I` must be in `index.all_ids()`.

  **Reconstruction Semantics**
  * Uncompressed data is returned directly (potentially promoting to `float32`).
  * LVQ compressed data is reconstructed using this highest precision possible. For two
    level datasets, boths levels will be used.
  * LeanVec datasets will reconstruct using the full-precision secondary dataset.

* Added an upgrade tool `pysvs.upgrader.upgrade` to upgrade the serialization layout of SVS
  objects.

## `libsvs` (C++)

### Changes

* Overhauled object loading. Context free classes should now accept a
  `svs::lib::ContextFreeLoadTable` and contextual classes should take a
  `svs::lib::LoadTable`.

  While most of the top level API remains unchanged, users are encouraged to look at the
  at the definitions of these classes in `include/svs/lib/saveload/load.h` to understand

  their capabilities and API.
* Added a new optional loading function `try_load -> svs::lib::Expected` which tries to load
  an object from a table and fails gracefully without an exception if it cannot.

  This API enables discovery and matching of previously serialized object, allowing
  implementation of the auto-loading functionality in `pysvs`.

* Added the following member functions to `pysvs::Vamana` and `pysvs::DynamicVamana`
  ```c++
  void reconstruct_at(svs::data::SimpleDataView<float> data, std::span<const uint64_t> ids);
  ```
  which will reconstruct the vector indices in `ids` into the destination `data`.

  See the description in the release notes for `pysvs` regarding the semantics of
  reconstruction.

* Type erased orchestrators may now be compiled with support for multiple query types.
  For example,
  ```c++
  auto index = svs::Vamana::assemble<svs::lib::Types<svs::Float16, float>>(...);
  ```
  will compile an orchestrator capable of processing queries of either 16-bit or 32-bit
  floating values. The old syntax of
  ```c++
  auto index = svs::Vamana:assemble<float>(...);
  ```
  is still supported and yields an index capable of only processing a single query type.

  The augmented methods are given below:
  * `svs::Vamana::assemble`
  * `svs::Vamana::build`
  * `svs::DynamicVamana::assemble`
  * `svs::DynamicVamana::build`
  * `svs::Flat::assemble`

## Object Serialization Changes

* The implementation of two-level LVQ has changed from bitwise extension to true cascaded
  application of scalar quantization. See the discussion on
  [this PR](https://github.com/intel/ScalableVectorSearch/pull/28).

  Consequently, previously saved two-level LVQ datasets have had their serialization version
  incremented from `v0.0.2` to `v0.0.3` and will need to be regenerated.

* The data structure `svsbenchmark::vamana::BuildJob` has been updated from `v0.0.3` to
  `v0.0.4`. This change is backwards compatible, but users of this class are encouraged to
  upgrade as soon as possible.

  1. This change drops the `search_window_size` array and adds a field `preset_parameters`
     which must be an array of `svs::index::vamana::VamanaSearchParameters`. This is done to
     provide more fine-grained control of preset search parameters (including split-buffer,
     visited-set, and prefetching) more inline with `svsbenchmark::vamana::SearchJob`.

     Version `v0.0.3` with `search_window_size` be compatible until the next minor version
     of SVS with the semantics of constructing a non-split-buffered
     `svs::index::vamana::VamanaSearchParameters` with no visited filter and no prefetching.

  2. Added a field `save_directory` into which the constructed index will be saved.
     This field can be left as the empty string to indicate that no saving is desired.

     The benchmarking framework will ensure that all requested save directories are unique
     and that the parents of the requested directories exist.

     Older serialization version will default to no saving.

  An example of the new format can be obtained by running
  ```
  ./svs_benchmark vamana_static_build --example
  ```

# SVS 0.0.3 Release Notes

Highlighted Features

* *Turbo LVQ*: A SIMD optimized layout for LVQ that can improve end-to-end search
  performance for LVQ-4 and LVQ-4x8 encoded datasets.

* *Split-buffer*: An optimization that separates the search window size used during greedy
  search from the actual search buffer capacity. For datasets that use reranking (two-level
  LVQ and LeanVec), this allows more neighbors to be passed to the reranking phase without
  increasing the time spent in greedy search.

* *[LeanVec](https://arxiv.org/abs/2312.16335)* dimensionality reduction is now included as
  an experimental feature!
  This two-level technique uses a linear transformation to generate a primary dataset with
  lower dimensionality than full precision vectors.
  The initial portion of a graph search is performed using this primary dataset, then uses
  the full precision secondary dataset to rerank candidates.
  Because of the reduced dimensionality, LeanVec can greatly accelerate index constructed
  for high-dimensional datasets.

  As an experimental feature, future changes to this API are expected.
  However, the implementation in this release is sufficient to enable experimenting with
  this technique on your own datasets!

New Dependencies

* [Intel(R) MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html):
  Required by LeanVec.
* [toml](https://github.com/uiri/toml): Required by testing infrastructure.

## `pysvs` (Python)

### Additions and Changes

* Added the `LeanVecLoader` class as a dataset loader enabling use of
  [LeanVec](https://arxiv.org/abs/2312.16335) dimensionality reduction.

  The main constructor is shown below:
  ```
  pysvs.LeanVecLoader(
      loader: pysvs.VectorDataLoader,
      leanvec_dims: int,
      primary: pysvs.LeanVecKind = pysvs.LeanVecKind.lvq8,
      secondary: pysvs.LeanVecKind = pysvs.LeanVecKind.lvq8
  )
  ```
  where:
  * `loader` is the loader for the uncompressed dataset.
  * `leanvec_dims` is the target reduced dimensionality of the primary dataset.
    This should be less than `loader.dims` to provide a performance boost.
  * `primary` is the encoding to use for the reduced-dimensionality dataset.
  * `secondary` is the encoding to use for the full-dimensionality dataset.

  Valid options for `pysvs.LeanVecKind` are: `float16, float32, lvq4, lvq8`.

  See the documentation for docstrings and an example.

* Search parameters controlling recall and performance for the Vamana index are now set and
  queried through a `pysvs.VamanaSearchParameters` configuration class. The layout of this
  class is as follows:
  ```
  class VamanaSearchParameters

  Parameters controlling recall and performance of the VamanaIndex.
  See also: `Vamana.search_parameters`.

  Attributes:
      buffer_config (`pysvs.SearchBufferConfig`, read/write): Configuration state for the
          underlying search buffer.
      search_buffer_visited_set (bool, read/write): Enable/disable status of the search
          buffer visited set.
  ```
  with `pysvs.SearchBufferConfig` defined by
  ```
  class pysvs.SearchBufferConfig

  Size configuration for the Vamana index search buffer.
  See also: `pysvs.VamanSearchParameters`, `pysvs.Vamana.search_parameters`.

  Attributes:
      search_window_size (int, read-only): The number of valid entries in the buffer
          that will be used to determine stopping conditions for graph search.
      search_buffer_capacity (int, read-only): The (expected) number of valid entries that
          will be available. Must be at least as large as `search_window_size`.
  ```
  Example usage is shown below.
  ```python
  index = pysvs.Vamana(...);
  # Get the current parameters of the index.
  parameters = index.search_parameters
  print(parameters)
  # Possible Output: VamanaSearchParameters(
  #    buffer_config = SearchBufferConfig(search_window_size = 0, total_capacity = 0),
  #    search_buffer_visited_set = false
  # )

  # Update our local copy of the search parameters
  parameters.buffer_config = pysvs.SearchBufferConfig(10, 20)
  # Assign the modified parameters to the index. Future searches will be affected.
  index.search_parameters = parameters
  ```

* Split search buffer for the Vamana search index. This is achieved by using different
  values for the `search_window_size` and `search_buffer_capacity` fields of the
  `pysvs.SearchBufferConfig` class described above.

  An index configured this way will maintain more entries in its search buffer while still
  terminating search relatively early. For implementation like two-level LVQ that use
  reranking, this can boost recall without significantly increasing the effective
  search window size.

  For uncompressed indexes that do not use reranking, split-buffer can be used to decrease
  the search window size lower than the requested number of neighbors (provided the
  capacity is at least the number of requested neighbors). This enables continued trading
  of recall for search performance.

* Added `pysvs.LVQStrategy` for picking between different flavors of LVQ. The values
  and meanings are given below.
  - `Auto`: Let pysvs decide from among the available options.
  - `Sequential`: Use the original implementation of LVQ which bit-packs subsequent vector
    elements sequentially in memory.
  - `Turbo`: Use an experimental implementation of LVQ that permutes the packing of
    subsequent vector elements to permit faster distance computations.

  The selection of strategy can be given using the `strategy` keyword argument of
  `pysvs.LVQLoader` and defaults to `pysvs.LVQStrategy.Auto`.

* Index construction and loading methods will now list the registered index specializations.

* Assigning the `padding` keyword to `LVQLoader` will now be respected when reloading a
  previously saved LVQ dataset.

* Changed the implementation of the greedy-search visited set to be effective when operating
  in the high-recall/high-neighbors regime. It can be enabled with:
  ```python
  index = pysvs.Vamana(...)
  p = index.search_parameters
  p.search_buffer_visited_set = True
  index.search_parameters = p
  ```

### Experimental Features

Features marked as *experimental* are subject to rapid API changes, improvement, and
removal.

* Added the `experimental_backend_string` read-only parameter to `pysvs.Vamana` to aid in
  recording and debugging the backend implementation.

* Introduced `pysvs.Vamana.experimental_calibrate` to aid in selecting the best runtime
  performance parameters for an index to achieve a desired recall.

  This feature can be used as follows:
  ```python
  # Create an index
  index = pysvs.Vamana(...)
  # Load queries and groundtruth
  queries = pysvs.read_vecs(...)
  groundtruth = pysvs.read_vecs(...)
  # Optimize the runtime state of the index for 0.90 10-recall-at-10
  index.experimental_calibrate(queries, groundtruth, 10, 0.90)
  ```
  See the documentation for a more detailed explanation.

### Deprecations

* Versions `0.0.1` and `0.0.2` of `VamanaConfigParameters` (the top-level configuration file
  for the Vamana index) are deprecated. The current version is now `v0.0.3`. Older versions
  will continue to work until the next minor release of SVS.

  To upgrade, use the `convert_legacy_vamana_index` binary utility described below.

* The attribute `pysvs.Vamana.visisted_set_enabled` is deprecated and will be removed in the
  next minor release of SVS. It is being replaced with `pysvs.Vamana.search_parameters`.

* The LVQ loader classes `pysvs.LVQ4`, `pysvs.LVQ8`, `pysvs.LVQ4x4`, `pysvs.LVQ4x8` and
  `pysvs.LVQ8x8` are deprecated in favor of a single class `pysvs.LVQLoader`. This class
  has similar arguments to the previous family, but encodes the number of bits for the
  primary and residual datasets as run-time values.

  For example,
  ```python
  # Old
  loader = pysvs.LVQ4x4("dataset", dims = 768, padding = 32)
  # New
  loader = pysvs.LVQLoader("dataset", primary = 4, residual = 4, dims = 768, padding = 32)
  ```

* Version `v0.0.2` of serialized LVQ datasets is *broken*, the current version is now
  `v0.0.3`. This change was made to facilitate a canonical on-disk representation of LVQ.

  Goind forward, previously saved LVQ formats can be reloaded using different runtime
  alignments and different packing strategies without requiring whole dataset recompression.

  Any previously saved datasets will need to be regenerated from uncompressed data.

### Build System Changes

Building `pysvs` using `cibuildwheel` now requires a custom docker container with Intel(R) MKL.
To build the container, run the following commands:
```sh
cd ./docker/x86_64/manylinux2014/
./build.sh
```

## `libsvs` (C++)

### Changes

* Added `svs::index::vamana::VamanaSearchParameters` and
  `svs::index::vamana::SearchBufferConfig`. The latter contains parameters for the search
  buffer sizing while the former groups all algorithmic and performance parameters of search
  together in a single class.
* API addition of `get_search_parameters()` and `set_search_parameters()` to `svs::Vamana`
  and `svs::DynamicVamana` as the new API for getting and setting all search parameters.
* Introducing split-buffer for the search buffer (see description in the Python section)
  to potentially increase recall when using reranking.
* Overhauled LVQ implementation, adding an additional template parameter to
  `lvq::CompressedVectorBase` and friends. This parameter assumes the following types:
  * `lvq::Sequential`: Store dimension encodings sequentially in memory. This corresponds
     to the original LVQ implementation.
  * `lvq::Turbo<size_t Lanes, size_t ElementsPerLane>`: Use a SIMD optimized format,
     optimized to use `Lanes` SIMD lanes, storing `ElementsPerLane`. Selection of these
     parameters requires some knowledge of the target hardware and appropriate overloads
     for decompression and distance computation.

     Accelerated methods require Intel(R) AVX-512 and are:
     * L2, IP, and decompression for LVQ 4 and LVQ 4x8 using `Turbo<16, 8>`
       (targeting Intel(R) AVX-512)
     * L2, IP, and decompression for LVQ 8 using `Turbo<16, 4>`.

* Added the following member function to `svs::lib::LoadContext`:
  ```c++
  /// Return the given relative path as a full path in the loading directory.
  std::filesystem::path LoadContext::resolve(const std::filesystem::path& relative) const;

  /// Return the relative path in `table` at position `key` as a full path.
  std::filesystem::path resolve(const toml::table& table, std::string_view key) const;
  ```

* Context-free saveable/loadable classes can now be saved/loaded directly from a TOML file
  without a custom directory using `svs::lib::save_to_file` and `svs::lib::load_from_file`.

* Distance functors can prevent missing `svs::distance::maybe_fix_arguments()` calls into
  hard errors by defining
  ```
  static constexpr bool must_fix_argument = true;
  ```
  in the class definition. Without this, `svs::distance::maybe_fix_argument()` will SFINAE
  away if a suitable `fix_argument()` member function is not found (the original behavior).

* The namespace `svs::lib::meta` has been removed. All entities previously defined there
  are now in `svs::lib`.

* Added a new Database file type. This file type will serve as a prototype for SSD-style
  data base files and is implemented in a way that can be extended by concrete
  implementations.

  This file has magic number `0x26b0644ab838c3a3` and contains a 16-byte UUID, 8-byte kind
  tag, and 24-byte version number. The 8-byte kind is the extension point that concrete
  implementations can use to define their own concrete implementations.

* Changed the implementation of the greedy search visited set to
  `svs::index::vamana::VisitedFilter`. This is a fuzzy associative data structure that may
  return false negatives (marking a neighbor as not visited when it has been visited) but
  has very fast lookups.

  When operating in the very high-recall/number of neighbors regime, enabling the visited
  set can yield performance improvements.

  It can be enabled with the following code:
  ```c++
  svs::Vamana index = /*initialize*/;
  auto p = index.get_search_parameters();
  p.search_buffer_visited_set(true);
  index.set_search_parameters(p);
  ```

### Deprecations

* The member functions `visited_set_enabled`, `enable_visited_set`, and
  `disable_visited_set` for `svs::Vamana` and `svs::DynamicVamana` are deprecated and will
  be removed in the next minor release of SVS.
* The class `svs::index::vamana::VamanaConfigParameters` has been renamed to
  `svs::index::vamana::VamanaIndexParameters` and its serialization version has been
  incremented to `v0.0.3`. Versions 0.0.1 and 0.0.2 will be compatible until the next minor
  release of SVS. Use the binary utility `convert_lebacy_vamana_index_config` to upgrade.
* Version `v0.0.2` of `svs::quantization::lvq::LVQDataset` has been upgraded to `v0.0.3` in
  a non-backward-compatible way. To facilitate a canonical on-disk representation of LVQ.

## Binary Utilities

* Added `convert_legacy_vamana_index_config` to upgrade Vamana index configuration file
  from version 0.0.1 or 0.0.2 to 0.0.3.

* Removed ``generate_vamana_config`` which created a Vamana index config file from extremely
  legacy formats.

## Testing

* Reference data for integration tests has been migrated to auto-generation from the
  benchmarking framework.

## Build System

The CMake variables were added.

* `SVS_EXPERIMENTAL_LEANVEC`: Enable LeanVec support, which requires Intel(R) MKL as a dependency.
  - Default (SVS, SVSBenchmark): `OFF`
  - Default (pysvs): `ON`

* `SVS_EXPERIMENTAL_CUSTOM_MKL`: Use Intel(R) MKL's custom shared object builder to create a minimal
  library to be installed with SVS. This enables relocatable builds to systems that do not
  have Intel(R) MKL installed and removes the need for Intel(R) MKL runtime environment variables.

  With this feature disabled, SVS builds against the system's Intel(R) MKL.

  - Default (SVS, SVSBenchmark): `OFF`
  - Default (pysvs): `ON`

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

