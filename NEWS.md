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

* [MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html):
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

Building `pysvs` using `cibuildwheel` now requires a custom docker container with MKL.
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

     Accelerated methods require AVX-512 and are:
     * L2, IP, and decompression for LVQ 4 and LVQ 4x8 using `Turbo<16, 8>`
       (targeting AVX 512)
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

* `SVS_EXPERIMENTAL_LEANVEC`: Enable LeanVec support, which requires MKL as a dependency.
  - Default (SVS, SVSBenchmark): `OFF`
  - Default (pysvs): `ON`

* `SVS_EXPERIMENTAL_CUSTOM_MKL`: Use MKL's custom shared object builder to create a minimal
  library to be installed with SVS. This enables relocatable builds to systems that do not
  have MKL installed and removes the need for MKL runtime environment variables.

  With this feature disabled, SVS builds against the system's MKL.

  - Default (SVS, SVSBenchmark): `OFF`
  - Default (pysvs): `ON`

