# SVS C API Design proposal

## Intro

This document contains SVS C API design proposal.

## Aspects and requirements

* Simplicity - minimal set of operations to create an index and use it
* Flexibility - let user to modify as many parameters and options as possible
  * Build parameters
  * Block sizes
  * Search parameters
  * Allocation
  * Thread pooling
* Error handling

## Concept

## Key abstractions

* Index
  * Initialized with non-empty dataset
  * TopK Search
  * Range search
  * Iterative search
  * Filtered search
* MutableIndex : Index
  * Initialized with a dataset and labels list
  * Add vectors with labels
  * Remove vectors by labels
  * Check if a label exists
  * Compute distance for label
  * Get vector by label
* Factory
  * Created with minimal parameters: index algorithm, mutable/immutable, dimensions, distance metric
  * Set storage configuration - default: Simple+FP32
  * Set algorithm configuration
  * Set custom threadpool - default: SVS native
  * Set custom allocator - default: SVS internal
  * Mutable only: set allocation blocking - default: SVS internal
* Index data storage cofiguration
  * Created with Kind: Simple, SQ, LVQ, LeanVec
  * Configuration "Simple":
    * Type: FP32, FP16, BF16
  * Configuration "SQ":
    * Signed/Unsigned
  * Configuration "LVQ":
    * Primary size: 4, 8
    * Residual size: 0, 4, 8
  * Configuration LeanVec:
    * Leanvec dimensions:
    * Primary storage configuration: FP32, FP16, BF16, LVQ4, LVQ8
    * Secondary storage configuration:  FP32, FP16, BF16, LVQ4, LVQ8
* Threadpool configuration
  * Threadpool kind: native, OMP, custom
  * SVS native: size
  * OMP: nothing (??size??)
  * Custom: `struct IThreadPool { size_t (*size)(void); void (*parallel_for)(void (*f)(size_t), size_t); };`
* Allocator configuration
  * Kind: simple, hugepage, custom
  * Custom allocator: `struct IAlloc { void* (*alloc)(size_t); void (*dealloc)(void*, size_t); };`

## Error handling

There is 2 possible ways to handle errors in C API

1. All API calls return status/error code when results are passed via "output argument" - "Intel oneDNN style"
2. All API calls return results when error handling is managed via "optional" pointer to status/error code variable - "OpenCL style"

Proposed API utilizes the second approach.

### Error details/message challenge

Possible ways to provide errors messages:

* Static map from error code to constant strings
  * Simple API: `const char* svs_error_message(svs_status_t);` 
  * Requires fine-grained error codes
  * Cost of code-to-message table maintainance
* Dynamically generated strings
  * Complicated API to manage dynamically allocated strings
  * Simplified error codes
  * Easy to add/update error messages
  * Allow to handle/explain object states (arguments, etc.) in messages

## API Sample

```c
// typedefs
int label_t;

// Error handling
typedef enum {
  svs_success;
  svs_invalid_argument;
  svs_out_of_memory;
  ...
} svs_status_t;

// Opaque types
svs_algorithm_t;
svs_storage_t;
svs_threadpool_t;
svs_allocator_t;

// Enums
enum svs_metric_t;
enum svs_type_t;

// Algorithm
typedef struct {...} svs_flat_params_t;
svs_algorithm_t svs_algo_create_flat(svs_flat_params_t* /*=NULL*/, svs_status_t* err_ret /*=NULL*/);

typedef struct {...} svs_vamana_params_t;
svs_algorithm_t svs_algo_create_vamana(svs_vamana_params_t* params /*=NULL*/, svs_status_t* err_ret /*=NULL*/);

svs_algorithm_t svs_algo_create_ivf(..., svs_status_t* err_ret /*=NULL*/);

// Storage
svs_storage_t svs_storage_create_simple(svs_type_t, svs_status_t* err_ret /*=NULL*/);
svs_storage_t svs_storage_create_sq(svs_type_t, svs_status_t* err_ret /*=NULL*/);
svs_storage_t svs_storage_create_lvq(svs_type_t primary, svs_type_t residual, svs_status_t* err_ret /*=NULL*/);
svs_storage_t svs_storage_create_leanvec(size_t leanvec_dims, svs_type_t primary, svs_type_t secondary, svs_status_t* err_ret /*=NULL*/);


// Threadpool
svs_threadpool_t svs_threadpool_create_native(size, svs_status_t* err_ret /*=NULL*/);
svs_threadpool_t svs_threadpool_create_omp(svs_status_t* err_ret /*=NULL*/);

typedef struct {
  void* ctx;
  size_t (*size)(void);
  void (*parallel_for)(void (*f)(size_t), size_t);
} svs_threadpool_i;
svs_threadpool_t svs_threadpool_create_custom(svs_threadpool_i*, svs_status_t* err_ret /*=NULL*/);

// Allocator
svs_allocator_t svs_allocator_create_simple(svs_status_t* err_ret /*=NULL*/);
svs_allocator_t svs_allocator_create_hugepage(svs_status_t* err_ret /*=NULL*/);

typedef struct svs_allocator_i_tag {
  void* ctx;
  void* (*alloc)(size_t);
  void (*dealloc)(void*, size_t);
} svs_allocator_i;
svs_allocator_t svs_allocator_create_custom(svs_allocator_i*, svs_status_t* err_ret /*=NULL*/);


// Factory
svs_factory_t svs_factory_create(svs_algorithm_t, size_t dims, svs_metric_t, svs_status_t* err_ret /*=NULL*/);
void svs_factory_set_storage(svs_factory_t, svs_storage_t, svs_status_t* err_ret /*=NULL*/);
void svs_factory_set_threadpool(svs_factory_t, svs_threadpool_t, svs_status_t* err_ret /*=NULL*/);
void svs_factory_set_allocator(svs_factory_t, svs_allocator_t, svs_status_t* err_ret /*=NULL*/);

// Index
svs_index_t svs_index_create(svs_factory_t, float* data, size_t size, svs_status_t* err_ret /*=NULL*/);
svs_index_t svs_index_create_dynamic(svs_factory_t, float* data, svs_label_t* labels, size_t size, size_t blocksize /*=0*/, svs_status_t* err_ret /*=NULL*/);

// Results
typedef struct {
  size_t queries_num;
  size_t* results_nums;
  svs_label_t* labels;
  float* distances;
} svs_results_t;

svs_results_t (*svs_results_allocator_i)(size_t queries_num, size_t* results_nums);

// Index queries
size_t svs_index_search_topk(svs_index_t, float* queries, size_t* queries_num, int k, svs_results_allocator_i results, svs_status_t* err_ret /*=NULL*/);
size_t svs_index_search_range(svs_index_t, float* queries, size_t* queries_num, float range, svs_results_allocator_i results, svs_status_t* err_ret /*=NULL*/);


```
