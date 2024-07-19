#
# Copyright (C) 2023-present, Intel Corporation
#
# You can redistribute and/or modify this software under the terms of the
# GNU Affero General Public License version 3.
#
# You should have received a copy of the GNU Affero General Public License
# version 3 along with this software. If not, see
# <https://www.gnu.org/licenses/agpl-3.0.en.html>.
#

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
test_leanvec_data_matrix = str(TEST_DATASET_DIR.joinpath("leanvec_data_matrix.fvecs"))
test_leanvec_query_matrix = str(TEST_DATASET_DIR.joinpath("leanvec_query_matrix.fvecs"))

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


def test_close_lvq(original, reconstructed, primary_bits: int, residual_bits: int = 0):
    """
    Test that the reconstructed values are within the expected tolerance for LVQ compressed
    data.

    Arguments:
        - original: The original, uncompressed data.
        - reconstucted: The reconstructed data.

    Keyword Arguments:
        - primary_bits: The number of bits in the primary encoding.
        - residual_bits: The number of bits in the residual encoding.
    """

    # Obtain the difference between the maximum and minimum values in the pre-processed
    # dataset.
    spans = svs.common.get_lvq_range(original)

    # Compute the max delta for each component of the dataset.
    # NOTE: We *should* divide by another factor of two here, but there are some values in
    # the LVQ quantization space that will exceed this threshold due to compression
    # limitations.
    #
    # See the C++ tests for LVQ reconstruction for a more complete explanation.
    deltas = spans / (((2 ** primary_bits) - 1) * 2)
    if residual_bits != 0:
        deltas = deltas / ((2 ** residual_bits) - 1)

    # Ensure that each reconstructed value is within the target threshold (plus a tiny
    # fudge factor to help offset rounding imprecision.
    upper_bound = np.expand_dims(deltas, axis = 1)
    upper_bound = upper_bound + 0.0125 * upper_bound;
    return np.all(np.abs(original - reconstructed) <= upper_bound)
