# Copyright 2025 Intel Corporation
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
def assert_equal(lhs, rhs, message: str = "", epsilon = 0.05):
    if DEBUG_MODE:
        print(f"{message}: {lhs} == {rhs}")
    else:
        assert lhs < rhs + epsilon, message
        assert lhs > rhs - epsilon, message

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
    svs.generate_test_dataset(
        1000,                       # Create 1000 vectors in the dataset.
        100,                        # Generate 100 query vectors.
        256,                        # Set the vector dimensionality to 256.
        test_data_dir,              # The directory where results will be generated.
        data_seed = 1234,           # Random number seed for reproducibility.
        query_seed = 5678,          # Random number seed for reproducibility.
        num_threads = 4,            # Number of threads to use.
        distance = svs.DistanceType.L2,   # The distance type to use.
    )
    # [generate-dataset]

    # [create-loader]
    # We are going to construct a LeanVec dataset on-the-fly from uncompressed data.
    # First, we construct a loader for the uncompressed data.
    uncompressed_loader = svs.VectorDataLoader(
        os.path.join(test_data_dir, "data.fvecs"),
        svs.DataType.float32
    )

    # Next - we construct a LVQLoader which is configured to use LVQ compression with 4 
    # bits for the primary and 8 bits for the residual quantization.
    B1 = 4    # Number of bits for the first level LVQ quantization
    B2 = 8    # Number of bits for the residuals quantization
    compressed_loader = svs.LVQLoader(uncompressed_loader,
        primary=B1,
        residual=B2,
    )
    # [create-loader]
    
    # An index can be constructed using a LeanVec dataset.
    # [build-parameters]    
    parameters = svs.VamanaBuildParameters(
        graph_max_degree = 64,
        window_size = 128,
    )
    # [build-parameters]

    # [build-index]
    index = svs.Vamana.build(
        parameters,
        compressed_loader,
        svs.DistanceType.L2,
        num_threads = 4,
    )
    # [build-index]

    # Set the search window size of the index and perform queries and load the queries.
    # [perform-queries]
    n_neighbors = 10    
    index.search_window_size = 20
    index.num_threads = 4    

    queries = svs.read_vecs(os.path.join(test_data_dir, "queries.fvecs"))    
    I, D = index.search(queries, n_neighbors)
    # [perform-queries]

    # Compare with the groundtruth.
    # [recall]
    groundtruth = svs.read_vecs(os.path.join(test_data_dir, "groundtruth.ivecs"))    
    recall = svs.k_recall_at(groundtruth, I, n_neighbors, n_neighbors)
    print(f"Recall = {recall}")    
    # [recall]
    assert_equal(recall, 0.953)

    # Finally, we can save the index and reload from a previously saved set of files.
    # [saving-loading]
    index.save(
        os.path.join(test_data_dir, "example_config"),
        os.path.join(test_data_dir, "example_graph"),
        os.path.join(test_data_dir, "example_data"),
    )
    
    index = svs.Vamana(
        os.path.join(test_data_dir, "example_config"),
        os.path.join(test_data_dir, "example_graph"),        
        os.path.join(test_data_dir, "example_data"),
        svs.DistanceType.L2,
        num_threads = 4,
    )
    # [saving-loading]


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
