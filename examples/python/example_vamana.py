# Import `unittest` to allow for automated testing.
import unittest

# [imports]
import os
import pysvs
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
        recall = pysvs.k_recall_at(groundtruth, I, 10, 10)
        assert_equal(
            recall, expected[window_size], f"Standard Search Check ({window_size})"
        )

def run_test_onelevel8(index, queries, groundtruth):
    expected = {
        10: 0.5676,
        20: 0.7405,
        30: 0.8266,
        40: 0.8804,
    }

    for window_size in range(10, 50, 10):
        index.search_window_size = window_size
        I, D = index.search(queries, 10)
        recall = pysvs.k_recall_at(groundtruth, I, 10, 10)
        assert_equal(
            recall, expected[window_size], f"Compressed Search Check ({window_size})"
        )

def run_test_build_onelevel8(index, queries, groundtruth):
    expected = {
        10: 0.5703,
        20: 0.7407,
        30: 0.8278,
        40: 0.8799,
    }

    for window_size in range(10, 50, 10):
        index.search_window_size = window_size
        I, D = index.search(queries, 10)
        recall = pysvs.k_recall_at(groundtruth, I, 10, 10)
        assert_equal(
            recall, expected[window_size], f"Compressed Search Check ({window_size})"
        )

# Shadow this as a global to make it available to the test-case clean-up.
test_data_dir = None

def run():
    # [generate-dataset]
    # Create a test dataset.
    # This will create a directory "pysvs_example_data" and populate it with three
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
    # Now, we can build a graph index over the data set.
    parameters = pysvs.VamanaBuildParameters(
        graph_max_degree = 64,
        window_size = 128,
    )
    # [build-parameters]

    # [build-index]
    # Build the index.
    index = pysvs.Vamana.build(
        parameters,
        pysvs.VectorDataLoader(
            os.path.join(test_data_dir, "data.fvecs"), pysvs.DataType.float32
        ),
        pysvs.DistanceType.L2,
        num_threads = 4,
    )
    # [build-index]

    # [build-index-fromNumpyArray]
    # Build the index.
    data = pysvs.read_vecs(os.path.join(test_data_dir, "data.fvecs"))
    index = pysvs.Vamana.build(
        parameters,
        data,
        pysvs.DistanceType.L2,
        num_threads = 4,
    )
    # [build-index-fromNumpyArray]

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
    assert(recall == 0.8288)
    # [perform-queries]

    # [search-window-size]
    # We can vary the search window size to demonstrate the trade off in accuracy.
    for window_size in range(10, 50, 10):
        index.search_window_size = window_size
        I, D = index.search(queries, 10)
        recall = pysvs.k_recall_at(groundtruth, I, 10, 10)
        print(f"Window size = {window_size}, Recall = {recall}")
    # [search-window-size]

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
    index = pysvs.Vamana(
        os.path.join(test_data_dir, "example_config"),
        pysvs.GraphLoader(os.path.join(test_data_dir, "example_graph")),
        pysvs.VectorDataLoader(
            os.path.join(test_data_dir, "example_data"), pysvs.DataType.float32
        ),
        pysvs.DistanceType.L2,
        num_threads = 4,
    )

    # We can rerun the queries to ensure everything works properly.
    index.search_window_size = 30
    I, D = index.search(queries, 10)

    # Compare with the groundtruth.
    recall = pysvs.k_recall_at(groundtruth, I, 10, 10)
    print(f"Recall = {recall}")
    assert(recall == 0.8288)
    # [loading]

    ##### Begin Test
    run_test_float(index, queries, groundtruth)
    ##### End Test

    # [only-loading]
    # We can reload an index from a previously saved set of files.
    index = pysvs.Vamana(
        os.path.join(test_data_dir, "example_config"),
        pysvs.GraphLoader(os.path.join(test_data_dir, "example_graph")),
        pysvs.VectorDataLoader(
            os.path.join(test_data_dir, "example_data"), pysvs.DataType.float32
        ),
        pysvs.DistanceType.L2,
        num_threads = 4,
    )
    # [only-loading]

    # [runtime-nthreads]
    index.num_threads = 4
    # [runtime-nthreads]

    #####
    ##### Quantization.
    #####

    # [search-compressed-loader]
    data_loader = pysvs.VectorDataLoader(
        os.path.join(test_data_dir, "example_data"),
        pysvs.DataType.float32,
        dims = 128    # Passing dim is NOT optional in this case
    )
    padding = 32
    compressed_loader = pysvs.LVQ8(
        data_loader,
        padding        # Passing padding is optional.
    )
    # [search-compressed-loader]

    # [search-compressed]
    index = pysvs.Vamana(
        os.path.join(test_data_dir, "example_config"),
        pysvs.GraphLoader(os.path.join(test_data_dir, "example_graph")),
        compressed_loader,
        # Optional keyword arguments
        distance = pysvs.DistanceType.L2,
        num_threads = 4
    )

    # Compare with the groundtruth..
    index.search_window_size = 30
    I, D = index.search(queries, 10)
    recall = pysvs.k_recall_at(groundtruth, I, 10, 10)
    print(f"Compressed recall: {recall}")
    assert(recall == 0.8266)
    # [search-compressed]

    ##### Begin Test
    run_test_onelevel8(index, queries, groundtruth)
    ##### End Test

    # [build-index-compressed]
    # Build the index.
    index = pysvs.Vamana.build(
        parameters,
        compressed_loader,
        pysvs.DistanceType.L2,
        num_threads = 4
    )
    # [build-index-compressed]

    # 1. Building Uncompressed
    # 2. Loading Uncompressed
    # 3. Loading with a recompressor

    # We can rerun the queries to ensure everything works properly.
    index.search_window_size = 30
    I, D = index.search(queries, 10)

    # Compare with the groundtruth.
    recall = pysvs.k_recall_at(groundtruth, I, 10, 10)
    print(f"Recall = {recall}")
    assert(recall == 0.8278)
    # [loading]

    ##### Begin Test
    run_test_build_onelevel8(index, queries, groundtruth)
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
