# SVS 0.5.0 Release Notes

## Additions and Changes

* New C API for SVS — index construction, search, and lifecycle management from C, including filtered TopK search, memory accounting (`get_memory_usage` / `get_memory_breakdown`), and index threadpool size control (#305, #306, #352, #354, #363, #370)

* Experimental fine-grain concurrent Vamana index, `svs::index::vamana::concurrent::MutableVamanaIndex`, supporting lock-free search concurrent with `add_points`, `delete_entries`, and `consolidate` (#369)

* LVQ and LeanVec dataset support for the fine-grain concurrent Vamana index

* `get_memory_usage()` added to VamanaIndex to report allocated bytes (#345)

* Optional `blocksize_elements` added to `BlockingParameters` (#344)

* `element_size()` added to LVQ and LeanVec datasets

* `CompressedDataset` instantiations added to the shared library

* Fixed LeanVec `is_pca` to derive centering from matrix equality rather than matrix presence

* Fixed GCC-12.x prefetch-loop collapse in `greedy_search` neighbor prefetch (#361)

* `Blocked` class refactored to meet allocator requirements (#351)
