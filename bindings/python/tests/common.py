# Copyright 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import time
import unittest
import numpy as np

import svs.common

# Logic for enumerating the paths to the test dataset.
# We start with the current file and go up to the root
# directory of the SVS project.
_current_file = Path(__file__).parent.resolve() #<prefix>/svs/bindings/python/tests
ROOT_DIR = _current_file.parents[2]
print("Root:", ROOT_DIR)
TEST_DATASET_DIR = ROOT_DIR.joinpath("data", "test_dataset")

# Main exports
test_data_svs = str(TEST_DATASET_DIR.joinpath("data_f32.svs"))
test_data_vecs = str(TEST_DATASET_DIR.joinpath("data_f32.fvecs"))
test_data_dims = 128
test_graph = str(TEST_DATASET_DIR.joinpath("graph_128.svs"))
test_vamana_config = str(TEST_DATASET_DIR.joinpath("vamana_config.toml"))
test_metadata = str(TEST_DATASET_DIR.joinpath("metadata.svs"))
test_queries = str(TEST_DATASET_DIR.joinpath("queries_f32.fvecs"))
test_groundtruth_l2 = str(TEST_DATASET_DIR.joinpath("groundtruth_euclidean.ivecs"))
test_groundtruth_mip = str(TEST_DATASET_DIR.joinpath("groundtruth_mip.ivecs"))
test_groundtruth_cosine = str(TEST_DATASET_DIR.joinpath("groundtruth_cosine.ivecs"))
test_vamana_reference = str(TEST_DATASET_DIR.joinpath("reference/vamana_reference.toml"))

test_number_of_vectors = 10000
test_dimensions = 128

#####
##### Helper Functions
#####

def isapprox(x, y, epsilon = 0.0001):
    """
    Return true if `abs(x - y) < epsilon`.
    """
    return abs(x - y) < epsilon

def timed(f, *args, iters = 10, ignore_first = False):
    """
    Perform `f(*args)` for `iters` iterations.
    Return a tuple `(ret, time)` where `ret` is the result of `f(*args)` and `time`
    is the total execution time.
    """
    if ignore_first:
        f(*args)

    tic = time.time()
    for i in range(iters):
        ret = f(*args)

    toc = time.time()
    return (ret, toc - tic)

def get_test_set(A, num_entries: int):
    """
    Return last 'num_entries' rows from the given two dimensional matrix A.
    We use initial subset of queries for training and the remaining for testing.
    This function is used to extract testing queries and groundtruths.
    """
    assert(A.ndim == 2)
    assert(A.shape[0] >= num_entries)
    return A[-num_entries:];

def test_threading(f, *args, validate = None, iters = 4, print_times = False):
    """
    Test that the threading portion of an index manager `f` is working correctly.
    Arguments:
        - `f`: The index manager under test.
        - `*args`: Arguments to pass to the search procedure of the manager.

    Keyword Arguments:
        - `validate`: A lambda function accepting the results of `f(*args)` and
            returning `True` if the results are correct and `False` otherwise.

            If no validation is desired, can be `None`.

        - `iters`: The number of times to run `f(*args)` to get the approximate run time.

        - `print_times`: Boolean indicating if the execution times of the single and
            multithreaded cases should be printed to `stdout`.
    """
    # In order to get helpful assertions, we need to instantiate a TestCase object.
    testcase = unittest.TestCase()


    # Single threaded base case.
    f.num_threads = 1
    testcase.assertEqual(f.num_threads, 1)
    result, base_time = timed(f.search, *args, iters = iters, ignore_first = True)
    if validate is not None:
        testcase.assertTrue(validate(result))

    # Multithreaded alternate case.
    f.num_threads = 2
    testcase.assertEqual(f.num_threads, 2)
    result, new_time = timed(f.search, *args, iters = iters, ignore_first = True)
    if validate is not None:
        testcase.assertTrue(validate(result))

    if print_times:
        print("Base Time:", base_time)
        print("Threaded Time:", new_time)

    # Not an exact measurement, but generally close.
    # For short lived processes, we generally see closer to a 3x speedup than a 4x
    # speedup when using 4 threads.
    testcase.assertTrue(1.3 * new_time < base_time)

# Similar test as GetDistanceTester for python
def test_get_distance(index, distance, data = svs.read_vecs(test_data_vecs), test_distance = True):
    """
    Test the get_distance method of an index by comparing its results with direct distance computation.

    Arguments:
        index: The SVS index object with get_distance method
    """
    # Skip get_distance_test if flag is set
    if not test_distance:
        return

    tolerance=1e-2
    query_id = 10
    index_id = 100
    dt = data.dtype
    query_vector_raw = np.array(data[query_id], dtype=dt)
    indexed_vector_raw = np.array(data[index_id], dtype=dt)
    index_distance  = index.get_distance(index_id, query_vector_raw)
    # Up cast to avoid overflow
    query_vector = query_vector_raw.astype(np.float32)
    indexed_vector = indexed_vector_raw.astype(np.float32)

    # Compute distance based on distance type
    if distance == svs.DistanceType.L2:
        expected_distance = np.sum((query_vector - indexed_vector) ** 2)
    elif distance == svs.DistanceType.MIP:
        expected_distance = np.dot(query_vector, indexed_vector)
    elif distance == svs.DistanceType.Cosine:
        qn = np.linalg.norm(query_vector)
        vn = np.linalg.norm(indexed_vector)
        if qn == 0 or vn == 0:
            expected_distance = 0.0
        else:
            expected_distance = (np.dot(query_vector, indexed_vector) / (qn * vn))
    else:
        raise ValueError(f"Unsupported DistanceType: {distance}")

    print(index_distance, expected_distance, distance, dt)
    assert abs(index_distance - expected_distance) < tolerance

    # Test out of bounds ID
    try:
        index.get_distance(index_id + 99999, query_vector_raw)
        assert False, "Should have exception"
    except Exception as e:
        pass
