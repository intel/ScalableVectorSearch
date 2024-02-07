# Import `unittest` to allow for auotmated testing.
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

test_data_dir = None

def run():
    # [generate-dataset]
    # Create a test dataset.
    # This will create a directory "example_data_vamana" and populate it with three
    # entries:
    # - data.fvecs: The test dataset.
    # - queries.fvecs: The test queries.
    # - groundtruth.fvecs: The groundtruth.
    test_data_dir = "./example_data_vamana"
    pysvs.generate_test_dataset(
        1000,                       # Create 1000 vectors in the dataset.
        100,                        # Generate 100 query vectors.
        256,                        # Set the vector dimensionality to 256.
        test_data_dir,              # The directory where results will be generated.
        data_seed = 1234,           # Random number seed for reproducibility.
        query_seed = 5678,          # Random number seed for reproducibility.
        num_threads = 4,            # Number of threads to use.
        distance = pysvs.DistanceType.MIP,   # The distance type to use.
    )
    # [generate-dataset]

    # [create-loader]
    # We are going to construct a LeanVec dataset on-the-fly from uncompressed data.
    # First, we construct a loader for the uncompressed data.
    uncompressed_loader = pysvs.VectorDataLoader(
        os.path.join(test_data_dir, "data.fvecs"),
        pysvs.DataType.float32
    )

    # Next - we construct a LeanVecLoader.
    # This loader is configured to perform the following:
    # - Reduce dimensionality of the primary dataset to 256 dimensions.
    # - Use LVQ8 for the primary dataset.
    # - Use Float16 for the secondary, unreduced dataset.
    leanvec_loader = pysvs.LeanVecLoader(
        uncompressed_loader,
        128,                                         # The reduced number of dimensions.
        primary_kind = pysvs.LeanVecKind.lvq8,       # The encoding of the primary dataset.
        secondary_kind = pysvs.LeanVecKind.float16,  # The encoding of the secondary dataset.
    )
    # [create-loader]

    # [build-and-search-index]
    # An index can be constructed using a LeanVec dataset.
    # Use an alpha less than 1 since we are using the Inner Product distance.
    parameters = pysvs.VamanaBuildParameters(
        alpha = 0.95,
        graph_max_degree = 64,
        prune_to = 60,
        window_size = 128,
    )

    index = pysvs.Vamana.build(
        parameters,
        leanvec_loader,
        pysvs.DistanceType.MIP,
        num_threads = 4,
    )

    # Load queries and ground-truth.
    queries = pysvs.read_vecs(os.path.join(test_data_dir, "queries.fvecs"))
    groundtruth = pysvs.read_vecs(os.path.join(test_data_dir, "groundtruth.ivecs"))

    # Set the search window size of the index and perform queries.
    p = index.search_parameters
    p.buffer_config = pysvs.SearchBufferConfig(30, 60)
    index.search_parameters = p
    I, D = index.search(queries, 10)

    # Compare with the groundtruth.
    recall = pysvs.k_recall_at(groundtruth, I, 10, 10)
    print(f"Recall = {recall}")
    assert_equal(recall, 0.863, "initial recall")
    # [build-and-search-index]

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
