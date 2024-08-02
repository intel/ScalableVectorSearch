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
