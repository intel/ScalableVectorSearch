# Import `unittest` to allow for automated testing.
import unittest

# [imports]
import os
import pysvs
import numpy as np
# [imports]

DEBUG_MODE = False
def assert_equal(lhs, rhs, message: str = ""):
    if DEBUG_MODE:
        print(f"{message}: {lhs} == {rhs}")
    else:
        assert lhs == rhs, message

def run_test_float(index, queries, groundtruth):
    expected = {
        10: 0.563,
        20: 0.729,
        30: 0.8202,
        40: 0.8751,
    }

    for window_size in range(10, 50, 10):
        index.search_window_size = window_size
        I, D = index.search(queries, 10)
        recall = pysvs.k_recall_at(groundtruth, I, 10, 10)
        assert_equal(
            recall, expected[window_size], f"Standard Search Check ({window_size})"
        )

# Shadow this as a global to make it available to the test-case clean-up.
test_data_dir = None

def run():
    # [generate-dataset]
    # Create a test dataset.
    # This will create a directory "example_data_vamana" and populate it with three
    # entries:
    # - data.fvecs: The test dataset.
    # - queries.fvecs: The test queries.
    # - groundtruth.ivecs: The groundtruth.
    test_data_dir = "./example_data_vamana"
    pysvs.generate_test_dataset(
        10000,                      # Create 10000 vectors in the dataset.
        1000,                       # Generate 1000 query vectors.
        128,                        # Set the vector dimensionality to 128.
        test_data_dir,              # The directory where results will be generated.
        data_seed = 1234,           # Random number seed for reproducibility.
        query_seed = 5678,          # Random number seed for reproducibility.
        num_threads = 4,            # Number of threads to use.
        distance = pysvs.DistanceType.L2,   # The distance type to use.
    )
    # [generate-dataset]

    # [build-parameters]
    parameters = pysvs.VamanaBuildParameters(
        graph_max_degree = 64,
        window_size = 128,
    )
    # [build-parameters]

    # [build-index]
    n = 9000
    data = pysvs.read_vecs(os.path.join(test_data_dir, "data.fvecs"))
    idx = np.arange(data.shape[0]).astype('uint64')

    index = pysvs.DynamicVamana.build(
        parameters,
        data[:n],
        idx[:n],
        pysvs.DistanceType.L2,
        num_threads = 4,
    )
    # [build-index]

    # [add-vectors]
    # Add the following 1000 vectors to the index.
    index.add(data[n:n+1000], idx[n:n+1000])
    # [add-vectors]

    # [remove-vectors]
    # Remove the first 100 vectors from the index.
    index.delete(idx[:100])
    # [remove-vectors]

    # [consolidate-index]
    # Consolidate the index.
    index.consolidate().compact(1000)
    # [consolidate-index]

    # [load-aux]
    # Load the queries and ground truth.
    queries = pysvs.read_vecs(os.path.join(test_data_dir, "queries.fvecs"))
    groundtruth = pysvs.read_vecs(os.path.join(test_data_dir, "groundtruth.ivecs"))
    # [load-aux]

    # [perform-queries]
    # Set the search window size of the index and perform queries.
    index.search_window_size = 30
    I, D = index.search(queries, 10)

    # Compare with the groundtruth.
    recall = pysvs.k_recall_at(groundtruth, I, 10, 10)
    print(f"Recall = {recall}")
    assert(recall == 0.8202)
    # [perform-queries]

    ##### Begin Test
    run_test_float(index, queries, groundtruth)
    ##### End Test

    # [saving-results]
    # Finally, we can save the results.
    index.save(
        os.path.join(test_data_dir, "example_config"),
        os.path.join(test_data_dir, "example_graph"),
        os.path.join(test_data_dir, "example_data"),
    )
    # [saving-results]

    #####
    ##### Loading from an existing index.
    #####

    # [loading]
    # We can reload an index from a previously saved set of files.
    index = pysvs.DynamicVamana(
        os.path.join(test_data_dir, "example_config"),
        pysvs.GraphLoader(os.path.join(test_data_dir, "example_graph")),
        pysvs.VectorDataLoader(
            os.path.join(test_data_dir, "example_data"), pysvs.DataType.float32
        ),
        pysvs.DistanceType.L2,
        num_threads = 4,
    )
    # [loading]

    # We can rerun the queries to ensure everything works properly.
    index.search_window_size = 30
    I, D = index.search(queries, 10)

    # Compare with the groundtruth.
    recall = pysvs.k_recall_at(groundtruth, I, 10, 10)
    print(f"Recall = {recall}")
    assert(recall == 0.8202)


    ##### Begin Test
    run_test_float(index, queries, groundtruth)
    ##### End Test


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