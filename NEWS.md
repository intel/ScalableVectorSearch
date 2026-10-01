# SVS 0.5.0 Release Notes

## Additions and Changes

* New C API for SVS — index construction, search, and lifecycle management from C, including filtered TopK search, LeanVec OOD training, memory accounting and estimation, custom allocators, and index threadpool size control (#305, #306, #352, #354, #358, #363, #364, #370, #380)

* Experimental fine-grain concurrent Vamana index, `svs::index::vamana::concurrent::MutableVamanaIndex`, supporting lock-free search concurrent with `add_points`, `delete_entries`, and `consolidate` (#369)

* LVQ and LeanVec dataset support for the fine-grain concurrent Vamana index

* `replace_external_id` added to `MutableVamanaIndex` to rename a vector's external ID without re-inserting its data (#383)

* `get_memory_usage()` added to VamanaIndex to report allocated bytes (#345)

* Optional `blocksize_elements` added to `BlockingParameters` (#344)

* `element_size()` added to LVQ and LeanVec datasets

* `CompressedDataset` instantiations added to the shared library

* Reduced peak memory of LeanVec index builds by sizing transform batches automatically

* Fixed LeanVec `is_pca` to derive centering from matrix equality rather than matrix presence

* Fixed GCC-12.x prefetch-loop collapse in `greedy_search` neighbor prefetch (#361)

* `Blocked` class refactored to meet allocator requirements (#351)
