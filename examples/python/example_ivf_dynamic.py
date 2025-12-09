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
Example: Dynamic IVF Index

This example demonstrates how to:
1. Build a dynamic IVF index from scratch
2. Add new vectors to the index
3. Remove vectors from the index
4. Search the index
5. Compute distances between queries and indexed vectors
6. Save and reload the index
"""

import os
import svs
import numpy as np

def main():
    print("=" * 80)
    print("Dynamic IVF Index Example")
    print("=" * 80)
    
    # [generate-dataset]
    # Create a test dataset with 10,000 vectors
    test_data_dir = "./example_data_ivf_dynamic"
    print(f"\n1. Generating test dataset in '{test_data_dir}'...")
    
    svs.generate_test_dataset(
        1000,                           # Create 1000 vectors in the dataset
        100,                            # Generate 100 query vectors
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
        num_centroids = 20,             # Number of clusters/centroids
        minibatch_size = 1000,          # Minibatch size for k-means
        num_iterations = 10,            # Number of k-means iterations
        is_hierarchical = True,         # Use hierarchical k-means
        training_fraction = 0.1,        # Fraction of data for training
        seed = 0xc0ffee,                # Random seed for clustering
    )
    print(f"   ✓ Configured {build_parameters.num_centroids} centroids")
    # [build-parameters]
    
    # [build-index]
    # Build the dynamic IVF index with initial vectors
    print("\n3. Building dynamic IVF index...")
    n = 900  # Use 900 vectors for initial index
    
    # Load the data and create IDs
    data = svs.read_vecs(os.path.join(test_data_dir, "data.fvecs"))
    ids = np.arange(data.shape[0]).astype('uint64')
    
    # Build the index
    index = svs.DynamicIVF.build(
        parameters = build_parameters,
        data = data[:n],
        ids = ids[:n],
        distance_type = svs.DistanceType.L2,
        num_index_threads = 4,
    )
    print(f"   ✓ Index built with {index.size} vectors")
    print(f"   ✓ Index dimensions: {index.dimensions}")
    # [build-index]
    
    # [add-vectors]
    # Add new vectors to the index
    print("\n4. Adding 100 new vectors to the index...")
    initial_size = index.size
    index.add(data[n:n+100], ids[n:n+100])
    print(f"   ✓ Index size: {initial_size} → {index.size}")
    # [add-vectors]
    
    # [search-before-delete]
    # Search before deletion
    print("\n5. Searching the index...")
    queries = svs.read_vecs(os.path.join(test_data_dir, "queries.fvecs"))
    groundtruth = svs.read_vecs(os.path.join(test_data_dir, "groundtruth.ivecs"))
    
    # Configure search parameters
    search_params = svs.IVFSearchParameters(
        n_probes = 10,      # Number of clusters to search
        k_reorder = 1.0     # Reorder factor
    )
    index.search_parameters = search_params
    
    # Perform search
    num_neighbors = 10
    I, D = index.search(queries, num_neighbors)
    recall = svs.k_recall_at(groundtruth, I, num_neighbors, num_neighbors)
    print(f"   ✓ Recall@{num_neighbors}: {recall:.4f}")
    # [search-before-delete]
    
    # [get-distance]
    # Compute distance between a query and a specific indexed vector
    print("\n6. Computing distances with get_distance()...")
    query_vector = queries[0]
    test_id = 100
    
    if index.has_id(test_id):
        distance = index.get_distance(test_id, query_vector)
        print(f"   ✓ Distance from query to vector {test_id}: {distance:.6f}")
    else:
        print(f"   ✗ Vector {test_id} not found in index")
    # [get-distance]
    
    # [remove-vectors]
    # Remove vectors from the index
    print("\n7. Removing the first 50 vectors...")
    ids_to_delete = ids[:50]
    num_deleted = index.delete(ids_to_delete)
    print(f"   ✓ Deleted {num_deleted} vectors")
    print(f"   ✓ Index size after deletion: {index.size}")
    
    # Verify vectors are deleted
    if not index.has_id(25):
        print(f"   ✓ Verified: Vector ID 25 no longer in index")
    # [remove-vectors]
    
    # [consolidate-index]
    # Consolidate and compact the index
    print("\n8. Consolidating and compacting the index...")
    index.consolidate().compact(1000)
    print(f"   ✓ Index consolidated and compacted")
    # [consolidate-index]
    
    # [search-after-modifications]
    # Search after modifications
    print("\n9. Searching after modifications...")
    I, D = index.search(queries, num_neighbors)
    recall = svs.k_recall_at(groundtruth, I, num_neighbors, num_neighbors)
    print(f"   ✓ Recall@{num_neighbors}: {recall:.4f}")
    # [search-after-modifications]
    
    # [tune-search-parameters]
    # Experiment with different search parameters
    print("\n10. Tuning search parameters...")
    for n_probes in [5, 10, 20, 30]:
        search_params.n_probes = n_probes
        index.search_parameters = search_params
        
        I, D = index.search(queries, num_neighbors)
        recall = svs.k_recall_at(groundtruth, I, num_neighbors, num_neighbors)
        print(f"    n_probes={n_probes:2d} → Recall@{num_neighbors}: {recall:.4f}")
    # [tune-search-parameters]
    
    # [save-index]
    # Save the index to disk
    print("\n11. Saving the index...")
    config_dir = os.path.join(test_data_dir, "saved_config")
    data_dir = os.path.join(test_data_dir, "saved_data")
    
    # Create directories if they don't exist
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    index.save(config_dir, data_dir)
    print(f"   ✓ Index saved to:")
    print(f"     Config: {config_dir}")
    print(f"     Data:   {data_dir}")
    # [save-index]
    
    # [load-index]
    # Note: DynamicIVF.load() is being implemented for easier reload
    # For now, the index has been successfully saved and can be accessed at:
    print("\n12. Index saved successfully!")
    print(f"   ✓ Config: {config_dir}")
    print(f"   ✓ Data:   {data_dir}")
    print(f"   Note: load() API coming soon for simplified reload")
    # [load-index]
    
    # [get-all-ids]
    # Inspect final index state
    print("\n13. Final index inspection...")
    all_ids = index.all_ids()
    print(f"   ✓ Index contains {len(all_ids)} unique IDs")
    print(f"   ✓ ID range: [{np.min(all_ids)}, {np.max(all_ids)}]")
    # [get-all-ids]
    
    print("\n" + "=" * 80)
    print("Dynamic IVF Example Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
