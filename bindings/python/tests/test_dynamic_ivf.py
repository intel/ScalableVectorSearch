# Copyright 2026 Intel Corporation
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

# Tests for the Dynamic IVF index with save/load auto-detection.
import unittest
import os
import numpy as np
from tempfile import TemporaryDirectory

import svs

# Local dependencies
from .common import \
    test_data_svs, \
    test_data_vecs, \
    test_data_dims, \
    test_queries, \
    test_groundtruth_l2, \
    test_number_of_vectors

from .dynamic import ReferenceDataset


class DynamicIVFTester(unittest.TestCase):
    """
    Test building, adding, deleting points from the dynamic IVF index.
    Tests include save/load with auto-detection for uncompressed data types.
    """

    def id_check(self, index, ids):
        """Check that the index contains exactly the given IDs."""
        # Check that every id in `ids` is in the index.
        for this_id in ids:
            self.assertTrue(index.has_id(this_id))

        # Check that every id in the index is in `ids`
        all_ids = index.all_ids()
        for this_id in all_ids:
            self.assertTrue(this_id in ids)

    def recall_check(
            self,
            index: svs.DynamicIVF,
            reference: ReferenceDataset,
            num_neighbors: int,
            expected_recall: float,
            recall_delta: float,
        ):
        """Check recall and test save/reload functionality with auto-detection."""
        gt = reference.ground_truth(num_neighbors)
        I, D = index.search(reference.queries, num_neighbors)
        recall = svs.k_recall_at(gt, I, num_neighbors, num_neighbors)
        print(f"    Recall: {recall}")
        self.assertTrue(recall < expected_recall + recall_delta)
        self.assertTrue(recall > expected_recall - recall_delta)

        # Make sure saving and reloading work with auto-detection.
        with TemporaryDirectory() as tempdir:
            configdir = os.path.join(tempdir, "config")
            datadir = os.path.join(tempdir, "data")
            index.save(configdir, datadir)

            # Load with auto-detection - should detect data type automatically
            reloaded = svs.DynamicIVF.load(
                config_directory = configdir,
                data_directory = datadir,
                distance = svs.DistanceType.L2,
                num_threads = 2,
            )

            # Set the same search parameters as the original index
            reloaded.search_parameters = index.search_parameters

            # Get recall with the same search parameters.
            I, D = reloaded.search(reference.queries, num_neighbors)
            reloaded_recall = svs.k_recall_at(gt, I, num_neighbors, num_neighbors)

            print(f"    Reloaded Recall: {reloaded_recall}")
            self.assertTrue(reloaded_recall < expected_recall + recall_delta)
            self.assertTrue(reloaded_recall > expected_recall - recall_delta)

    def _build_clustering(self, data_loader, num_threads):
        """Build IVF clustering from a data loader."""
        build_params = svs.IVFBuildParameters(
            num_centroids = 64,
            minibatch_size = 128,
            num_iterations = 10,
            is_hierarchical = False,
            training_fraction = 0.8,
            hierarchical_level1_clusters = 0,
            seed = 42,
        )

        return svs.Clustering.build(
            build_parameters = build_params,
            data_loader = data_loader,
            distance = svs.DistanceType.L2,
            num_threads = num_threads,
        )

    def test_uncompressed(self):
        """Test DynamicIVF with uncompressed float32 data and auto-detection on reload."""
        num_threads = 2
        num_neighbors = 10
        expected_recall = 0.65
        expected_recall_delta = 0.20

        reference = ReferenceDataset(num_threads = num_threads)
        data, ids = reference.new_ids(5000)

        with TemporaryDirectory() as tempdir:
            data_file = os.path.join(tempdir, "data.fvecs")
            svs.write_vecs(data, data_file)

            data_loader = svs.VectorDataLoader(
                data_file,
                svs.DataType.float32,
                dims = data.shape[1]
            )

            clustering = self._build_clustering(data_loader, num_threads)

            # Assemble DynamicIVF from clustering
            index = svs.DynamicIVF.assemble_from_clustering(
                clustering = clustering,
                data_loader = data_loader,
                ids = ids,
                distance = svs.DistanceType.L2,
                num_threads = num_threads,
            )

            print(f"Testing uncompressed: {index.experimental_backend_string}")

            # Set search parameters
            search_params = svs.IVFSearchParameters(n_probes = 20, k_reorder = 100)
            index.search_parameters = search_params

            # Perform an ID check
            self.id_check(index, reference.ids())

            # Groundtruth Check with save/reload auto-detection
            print("Initial uncompressed")
            self.recall_check(
                index, reference, num_neighbors, expected_recall, expected_recall_delta
            )

            # Add and delete some vectors
            (add_data, add_ids) = reference.new_ids(1000)
            index.add(add_data, add_ids)
            print("After add")
            self.id_check(index, reference.ids())
            self.recall_check(
                index, reference, num_neighbors, expected_recall, expected_recall_delta
            )

            delete_ids = reference.remove_ids(1000)
            index.delete(delete_ids)
            print("After delete")
            self.id_check(index, reference.ids())
            self.recall_check(
                index, reference, num_neighbors, expected_recall, expected_recall_delta
            )

    def test_build_from_loader(self):
        """Test building DynamicIVF using a VectorDataLoader and explicit IDs."""
        num_threads = 2

        loader = svs.VectorDataLoader(test_data_svs, svs.DataType.float32, dims = test_data_dims)

        # Sequential IDs
        ids = np.arange(test_number_of_vectors, dtype = np.uint64)

        # Build IVF clustering
        build_params = svs.IVFBuildParameters(
            num_centroids = 128,
            minibatch_size = 128,
            num_iterations = 10,
            is_hierarchical = False,
            training_fraction = 0.8,
            hierarchical_level1_clusters = 0,
            seed = 42,
        )

        clustering = svs.Clustering.build(
            build_parameters = build_params,
            data_loader = loader,
            distance = svs.DistanceType.L2,
            num_threads = num_threads,
        )

        # Assemble DynamicIVF from clustering
        index = svs.DynamicIVF.assemble_from_clustering(
            clustering = clustering,
            data_loader = loader,
            ids = ids,
            distance = svs.DistanceType.L2,
            num_threads = num_threads,
        )

        # Basic invariants
        self.assertEqual(index.size, test_number_of_vectors)
        self.assertEqual(index.dimensions, test_data_dims)
        self.assertTrue(index.has_id(0))
        self.assertTrue(index.has_id(test_number_of_vectors - 1))

        # Search test
        queries = svs.read_vecs(test_queries)
        groundtruth = svs.read_vecs(test_groundtruth_l2)
        k = 10

        search_params = svs.IVFSearchParameters(n_probes = 30, k_reorder = 200)
        index.search_parameters = search_params

        I, D = index.search(queries, k)
        self.assertEqual(I.shape[1], k)
        recall = svs.k_recall_at(groundtruth, I, k, k)
        print(f"Build from loader recall: {recall}")
        self.assertTrue(0.5 < recall <= 1.0)

        # Test save and load with auto-detection
        with TemporaryDirectory() as tempdir:
            configdir = os.path.join(tempdir, "config")
            datadir = os.path.join(tempdir, "data")
            index.save(configdir, datadir)

            # Reload from saved directories - auto-detect data type
            reloaded = svs.DynamicIVF.load(
                config_directory = configdir,
                data_directory = datadir,
                distance = svs.DistanceType.L2,
                num_threads = num_threads
            )

            self.assertEqual(reloaded.size, test_number_of_vectors)
            self.assertEqual(reloaded.dimensions, test_data_dims)

            # Set search parameters and verify recall
            reloaded.search_parameters = search_params
            I, D = reloaded.search(queries, k)
            reloaded_recall = svs.k_recall_at(groundtruth, I, k, k)
            print(f"Reloaded recall: {reloaded_recall}")
            self.assertTrue(0.5 < reloaded_recall <= 1.0)
