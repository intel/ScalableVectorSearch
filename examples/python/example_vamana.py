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

# Import `unittest` to allow for automated testing.
import unittest

# [imports]
import os
import svs
# [imports]

DEBUG_MODE = False
def assert_equal(lhs, rhs, message: str = ""):
    if DEBUG_MODE:
        print(f"{message}: {lhs} == {rhs}")
    else:
        assert lhs == rhs, message

def run_test_float(index, queries, groundtruth):
    expected = {
        10: 0.5664,
        20: 0.7397,
        30: 0.8288,
        40: 0.8837,
    }

    for window_size in range(10, 50, 10):
        index.search_window_size = window_size
        I, D = index.search(queries, 10)
        recall = svs.k_recall_at(groundtruth, I, 10, 10)
        assert_equal(
            recall, expected[window_size], f"Standard Search Check ({window_size})"
        )

def run_test_two_level4_8(index, queries, groundtruth):
    expected = {
        10: 0.5482,
        20: 0.7294,
        30: 0.8223,
        40: 0.8756,
    }

    for window_size in range(10, 50, 10):
        index.search_window_size = window_size
        I, D = index.search(queries, 10)
        recall = svs.k_recall_at(groundtruth, I, 10, 10)
        assert_equal(
            recall, expected[window_size], f"Compressed Search Check ({window_size})"
        )

def run_test_build_two_level4_8(index, queries, groundtruth):
    expected = {
        10: 0.5484,
        20: 0.7295,
        30: 0.8221,
        40: 0.8758,
    }

    for window_size in range(10, 50, 10):
        index.search_window_size = window_size
        I, D = index.search(queries, 10)
        recall = svs.k_recall_at(groundtruth, I, 10, 10)
        assert_equal(
            recall, expected[window_size], f"Compressed Search Check ({window_size})"
        )

# Shadow this as a global to make it available to the test-case clean-up.
test_data_dir = None

def run():

    # ###
    # Generating test data
    # ###

    # [generate-dataset]
    # Create a test dataset.
    # This will create a directory "example_data_vamana" and populate it with three
    # entries:
    # - data.fvecs: The test dataset.
    # - queries.fvecs: The test queries.
    # - groundtruth.ivecs: The groundtruth.
    test_data_dir = "./example_data_vamana"
    svs.generate_test_dataset(
        10000,                      # Create 10000 vectors in the dataset.
        1000,                       # Generate 1000 query vectors.
        128,                        # Set the vector dimensionality to 128.
        test_data_dir,              # The directory where results will be generated.
        data_seed = 1234,           # Random number seed for reproducibility.
        query_seed = 5678,          # Random number seed for reproducibility.
        num_threads = 4,            # Number of threads to use.
        distance = svs.DistanceType.L2,   # The distance type to use.
    )
    # [generate-dataset]


    # ###
    # Building the index
    # ###

    # [build-parameters]
    # Now, we can build a graph index over the data set.
    parameters = svs.VamanaBuildParameters(
        graph_max_degree = 64,
        window_size = 128,
    )
    # [build-parameters]

    # [build-index]
    # Build the index.
    index = svs.Vamana.build(
        parameters,
        svs.VectorDataLoader(
            os.path.join(test_data_dir, "data.fvecs"), svs.DataType.float32
        ),
        svs.DistanceType.L2,
        num_threads = 4,
    )
    # [build-index]

    # [build-index-fromNumpyArray]
    # Build the index.
    data = svs.read_vecs(os.path.join(test_data_dir, "data.fvecs"))
    index = svs.Vamana.build(
        parameters,
        data,
        svs.DistanceType.L2,
        num_threads = 4,
    )
    # [build-index-fromNumpyArray]


    # ###
    # Searching the index
    # ###

    # [load-aux]
    # Load the queries and ground truth.
    queries = svs.read_vecs(os.path.join(test_data_dir, "queries.fvecs"))
    groundtruth = svs.read_vecs(os.path.join(test_data_dir, "groundtruth.ivecs"))
    # [load-aux]

    # [perform-queries]
    # Set the search window size of the index and perform queries.
    index.search_window_size = 30
    I, D = index.search(queries, 10)

    # Compare with the groundtruth.
    recall = svs.k_recall_at(groundtruth, I, 10, 10)
    print(f"Recall = {recall}")
    assert(recall == 0.8288)
    # [perform-queries]

    # [search-window-size]
    # We can vary the search window size to demonstrate the trade off in accuracy.
    for window_size in range(10, 50, 10):
        index.search_window_size = window_size
        I, D = index.search(queries, 10)
        recall = svs.k_recall_at(groundtruth, I, 10, 10)
        print(f"Window size = {window_size}, Recall = {recall}")
    # [search-window-size]

    ##### Begin Test
    run_test_float(index, queries, groundtruth)
    ##### End Test


    # ###
    # Saving the index
    # ###

    # [saving-results]
    # Finally, we can save the results.
    index.save(
        os.path.join(test_data_dir, "example_config"),
        os.path.join(test_data_dir, "example_graph"),
        os.path.join(test_data_dir, "example_data"),
    )
    # [saving-results]


    # ###
    # Reloading a saved index
    # ###

    # [loading]
    # We can reload an index from a previously saved set of files.
    index = svs.Vamana(
        os.path.join(test_data_dir, "example_config"),
        svs.GraphLoader(os.path.join(test_data_dir, "example_graph")),
        svs.VectorDataLoader(
            os.path.join(test_data_dir, "example_data"), svs.DataType.float32
        ),
        svs.DistanceType.L2,
        num_threads = 4,
    )

    # We can rerun the queries to ensure everything works properly.
    index.search_window_size = 30
    I, D = index.search(queries, 10)

    # Compare with the groundtruth.
    recall = svs.k_recall_at(groundtruth, I, 10, 10)
    print(f"Recall = {recall}")
    assert(recall == 0.8288)
    # [loading]

    ##### Begin Test
    run_test_float(index, queries, groundtruth)
    ##### End Test

    # [only-loading]
    # We can reload an index from a previously saved set of files.
    index = svs.Vamana(
        os.path.join(test_data_dir, "example_config"),
        svs.GraphLoader(os.path.join(test_data_dir, "example_graph")),
        svs.VectorDataLoader(
            os.path.join(test_data_dir, "example_data"), svs.DataType.float32
        ),
        svs.DistanceType.L2,
        num_threads = 4,
    )
    # [only-loading]

    # [runtime-nthreads]
    index.num_threads = 4
    # [runtime-nthreads]

#####
##### Main Executable
#####

if __name__ == "__main__":
    run()

#####
##### As a unit test.
#####

class VamanaExampleTestCase(unittest.TestCase):
    def tearDown(self):
        if test_data_dir is not None:
            print(f"Removing temporary directory {test_data_dir}")
            os.rmdir(test_data_dir)

    def test_all(self):
        run()
