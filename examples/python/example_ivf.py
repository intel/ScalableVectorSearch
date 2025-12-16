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

"""
Example: Static IVF Index

This example demonstrates how to:
1. Build clustering for IVF index
2. Assemble an IVF index from clustering
3. Search the index
4. Save and reload clustering
5. Load index from saved clustering
"""

import os
import svs
import numpy as np

def main():
    print("=" * 80)
    print("Static IVF Index Example")
    print("=" * 80)

    # [generate-dataset]
    # Create a test dataset
    test_data_dir = "./example_data_ivf"
    print(f"\n1. Generating test dataset in '{test_data_dir}'...")

    svs.generate_test_dataset(
        10000,                          # Create 10,000 vectors in the dataset
        1000,                           # Generate 1,000 query vectors
        128,                            # Set vector dimensionality to 128
        test_data_dir,                  # Directory where results will be generated
        data_seed = 1234,               # Random seed for reproducibility
        query_seed = 5678,              # Random seed for reproducibility
        num_threads = 4,                # Number of threads to use
        distance = svs.DistanceType.L2, # Distance metric
    )
    print("   ✓ Dataset generated")
    # [generate-dataset]

    # [build-parameters]
    # Configure clustering parameters for IVF
    print("\n2. Configuring build parameters...")
    build_parameters = svs.IVFBuildParameters(
        num_centroids = 50,             # Number of clusters/centroids
        minibatch_size = 2000,          # Minibatch size for k-means
        num_iterations = 20,            # Number of k-means iterations
        is_hierarchical = True,         # Use hierarchical k-means
        training_fraction = 0.5,        # Fraction of data for training
        seed = 0xc0ffee,                # Random seed for clustering
    )
    print(f"   ✓ Configured {build_parameters.num_centroids} centroids")
    # [build-parameters]

    # [load-data]
    # Load the dataset
    print("\n3. Loading dataset...")
    data_path = os.path.join(test_data_dir, "data.fvecs")
    data_loader = svs.VectorDataLoader(
        data_path,
        svs.DataType.float32,
        dims = 128
    )
    print(f"   ✓ Data loader created")
    # [load-data]

    # [build-clustering]
    # Build the clustering
    print("\n4. Building clustering (k-means)...")
    clustering = svs.Clustering.build(
        build_parameters = build_parameters,
        data_loader = data_loader,
        distance = svs.DistanceType.L2,
        num_threads = 4,
    )
    print(f"   ✓ Clustering built with {build_parameters.num_centroids} centroids")
    # [build-clustering]

    # [assemble-index]
    # Assemble the IVF index from clustering
    print("\n5. Assembling IVF index from clustering...")
    index = svs.IVF.assemble_from_clustering(
        clustering = clustering,
        data_loader = data_loader,
        distance = svs.DistanceType.L2,
        num_threads = 4,
        intra_query_threads = 1,
    )
    print(f"   ✓ Index assembled with {index.size} vectors")
    print(f"   ✓ Index dimensions: {index.dimensions}")
    # [assemble-index]

    # [configure-search]
    # Configure search parameters
    print("\n6. Configuring search parameters...")
    search_params = svs.IVFSearchParameters(
        n_probes = 10,      # Number of clusters to search
        k_reorder = 1.0     # Reorder factor (1.0 = no reordering)
    )
    index.search_parameters = search_params
    print(f"   ✓ Search parameters: n_probes={search_params.n_probes}")
    # [configure-search]

    # [search]
    # Perform search
    print("\n7. Searching the index...")
    queries = svs.read_vecs(os.path.join(test_data_dir, "queries.fvecs"))
    groundtruth = svs.read_vecs(os.path.join(test_data_dir, "groundtruth.ivecs"))

    num_neighbors = 10
    I, D = index.search(queries, num_neighbors)
    recall = svs.k_recall_at(groundtruth, I, num_neighbors, num_neighbors)
    print(f"   ✓ Recall@{num_neighbors}: {recall:.4f}")
    print(f"   ✓ Result shape: {I.shape}")
    # [search]

    # [save-clustering]
    # Save the clustering for later use
    print("\n8. Saving clustering...")
    clustering_path = os.path.join(test_data_dir, "clustering")
    clustering.save(clustering_path)
    print(f"   ✓ Clustering saved to '{clustering_path}'")
    # [save-clustering]

    # [load-and-assemble]
    # Load clustering and assemble a new index
    print("\n9. Loading clustering and assembling new index...")
    loaded_clustering = svs.Clustering.load_clustering(clustering_path)

    new_index = svs.IVF.assemble_from_clustering(
        clustering = loaded_clustering,
        data_loader = data_loader,
        distance = svs.DistanceType.L2,
        num_threads = 4,
        intra_query_threads = 1,
    )
    print(f"   ✓ New index assembled with {new_index.size} vectors")
    # [load-and-assemble]

    # [assemble-from-file]
    # Or directly assemble from file
    print("\n10. Assembling index directly from clustering file...")
    index_from_file = svs.IVF.assemble_from_file(
        clustering_path = clustering_path,
        data_loader = data_loader,
        distance = svs.DistanceType.L2,
        num_threads = 4,
        intra_query_threads = 1,
    )
    print(f"   ✓ Index assembled with {index_from_file.size} vectors")
    # [assemble-from-file]

    # [search-verification]
    # Verify both indices produce the same results
    print("\n11. Verifying search results consistency...")
    index_from_file.search_parameters = search_params
    I2, D2 = index_from_file.search(queries, num_neighbors)
    recall2 = svs.k_recall_at(groundtruth, I2, num_neighbors, num_neighbors)
    print(f"   ✓ Recall@{num_neighbors}: {recall2:.4f}")

    if np.allclose(D, D2):
        print("   ✓ Both indices produce identical results")
    else:
        print("   ✗ Warning: Results differ slightly (expected due to floating point)")
    # [search-verification]

    # [tune-search-parameters]
    # Experiment with different search parameters
    print("\n12. Tuning search parameters...")
    for n_probes in [5, 10, 20]:
        search_params.n_probes = n_probes
        index.search_parameters = search_params
        I_tuned, _ = index.search(queries, num_neighbors)
        recall_tuned = svs.k_recall_at(groundtruth, I_tuned, num_neighbors, num_neighbors)
        print(f"   ✓ n_probes={n_probes:2d}: Recall@{num_neighbors} = {recall_tuned:.4f}")
    # [tune-search-parameters]

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
