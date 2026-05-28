# SVS 0.4.0 Release Notes

## Additions and Changes

* Native serialization support for Indexes (#300)

* Clang-21 compiler support with gcc-11 headers for shared library builds

* Amazonlinux2 support for shared/static library dropped

* Consolidated to publish only minimal necessary shared library (14 -> 6) and runtime bindings (2 -> 1) release artifacts

* Adaptive batch size heuristic for filtered search in Vamana, with new `filter_stop` early-exit parameter (#309)

* IVF runtime support for `IDFilter`-based filtered search and dynamic `set_intra_query_threads` adjustment after `train()`/`load()` (#322)

* IVF inter-query thread pool initialization now validates and clamps thread count against the number of centroids (#329)

* In-memory (mmap) stream index mapping with new `map_to_file` and `map_to_memory` APIs on `FlatIndex` and `VamanaIndex` (#310)

* Add get_distance and reconstruct_at to VamanaIndex API (#315)
