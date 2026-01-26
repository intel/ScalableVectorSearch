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

# Tests for the IVF index portion of the SVS module.
import unittest
import os
import warnings
import toml

import numpy as np

from tempfile import TemporaryDirectory

import svs

# Local dependencies
from .common import \
    isapprox, \
    test_data_svs, \
    test_data_vecs, \
    test_data_dims, \
    test_queries, \
    test_groundtruth_l2, \
    test_groundtruth_mip, \
    test_groundtruth_cosine, \
    test_ivf_reference, \
    test_ivf_clustering, \
    test_number_of_vectors, \
    test_dimensions, \
    timed, \
    get_test_set, \
    test_get_distance

from .dataset import UncompressedMatcher

DEBUG = False

class IVFTester(unittest.TestCase):
    """
    Test IVF index querying, building, and saving.

    NOTE: The structure of these tests closely follows the integration tests in the C++
    library. Configurations and recalls values are used from the common reference file created
    using the benchmarking infrastructure
    """
    def setUp(self):
        # Initialize expected results from the common reference file
        with open(test_ivf_reference) as f:
            self.reference_results = toml.load(f)

    def _setup(self, loader: svs.VectorDataLoader):
        self.loader_and_matcher = [
            (loader, UncompressedMatcher("float32")),
        ]

    def _distance_map(self):
        return {
            svs.DistanceType.L2: "L2",
            svs.DistanceType.MIP: "MIP",
            svs.DistanceType.Cosine: "Cosine",
        }

    def _get_config_and_recall(self, test_type, distance, matcher):
        r = []
        for results in self.reference_results[test_type]:
            if (results['distance'] == distance) and matcher.is_match(results['dataset']):
                r.append(results['config_and_recall'])

        assert len(r) == 1, "Should match one results entry!"
        return r[0]

    def _parse_config_and_recall(self, results):
        params = results['search_parameters']
        n_probes = params['n_probes']
        k_reorder = params['k_reorder']
        k = results['num_neighbors']
        nq = results['num_queries']
        recall = results['recall']
        return n_probes, k_reorder, k, nq, recall

    def _get_build_parameters(self, test_type, distance, matcher):
        params = []
        for results in self.reference_results[test_type]:
            if (results['distance'] == distance) and matcher.is_match(results['dataset']):
                params.append(results['build_parameters'])

        assert len(params) == 1, "Should match one parameters entry!"
        params = params[0]

        return svs.IVFBuildParameters(
            num_centroids = params["num_centroids"],
            minibatch_size = params["minibatch_size"],
            num_iterations = params["num_iterations"],
            is_hierarchical = params["is_hierarchical"],
            training_fraction = params["training_fraction"],
            hierarchical_level1_clusters = params["hierarchical_level1_clusters"],
            seed = params["seed"],
        )

    def _test_single_query(
            self,
            ivf: svs.IVF,
            queries
        ):

        I_full, D_full = ivf.search(queries, 10)

        I_single = []
        D_single = []
        for i in range(queries.shape[0]):
            query = queries[i, :]
            self.assertTrue(query.ndim == 1)
            I, D = ivf.search(query, 10)

            self.assertTrue(I.ndim == 2)
            self.assertTrue(D.ndim == 2)
            self.assertTrue(I.shape == (1, 10))
            self.assertTrue(D.shape == (1, 10))

            I_single.append(I)
            D_single.append(D)

        I_single_concat = np.concatenate(I_single, axis = 0)
        D_single_concat = np.concatenate(D_single, axis = 0)
        self.assertTrue(np.array_equal(I_full, I_single_concat))
        self.assertTrue(np.array_equal(D_full, D_single_concat))

        # Throw an error on 3-dimensional inputs.
        queries_3d = queries[:, :, np.newaxis]
        with self.assertRaises(Exception) as context:
            ivf.search(queries_3d, 10)

        self.assertTrue("only accept numpy vectors or matrices" in str(context.exception))

    def _test_basic_inner(
            self,
            ivf: svs.IVF,
            matcher,
            num_threads: int,
            skip_thread_test: bool = False,
            test_single_query: bool = False,
        ):
        # Make sure that the number of threads is propagated correctly.
        self.assertEqual(ivf.num_threads, num_threads)

        # load the queries and groundtruth
        queries = svs.read_vecs(test_queries)
        groundtruth = svs.read_vecs(test_groundtruth_l2)

        self.assertEqual(queries.shape, (1000, 128))
        self.assertEqual(groundtruth.shape, (1000, 100))

        # Test get_distance
        data = svs.read_vecs(test_data_vecs)
        test_get_distance(ivf, svs.DistanceType.L2, data)

        # Data interface
        self.assertEqual(ivf.size, test_number_of_vectors)

        # The dimensionality exposed by the index should always match the original
        # dataset dimensions.
        self.assertEqual(ivf.dimensions, test_dimensions)

        expected_results = self._get_config_and_recall('ivf_test_search', 'L2', matcher)
        for expected in expected_results:
            n_probes, k_reorder, k, nq, expected_recall = \
                self._parse_config_and_recall(expected)

            parameters = svs.IVFSearchParameters(n_probes, k_reorder)
            ivf.search_parameters = parameters
            self.assertEqual(ivf.search_parameters.n_probes, n_probes)
            self.assertEqual(ivf.search_parameters.k_reorder, k_reorder)

            results = ivf.search(get_test_set(queries, nq), k)
            recall = svs.k_recall_at(get_test_set(groundtruth, nq), results[0], k, k)
            print(f"Recall = {recall}, Expected = {expected_recall}")
            if not DEBUG:
                self.assertTrue(isapprox(recall, expected_recall, epsilon = 0.0005))

        if test_single_query:
            self._test_single_query(ivf, queries)

    def _test_basic(self, loader, matcher, test_single_query: bool = False):
        num_threads = 2
        print("Assemble from file")
        ivf = svs.IVF.assemble_from_file(
            clustering_path = test_ivf_clustering,
            data_loader = loader,
            distance = svs.DistanceType.L2,
            num_threads = num_threads
        )

        print(f"Testing: {ivf.experimental_backend_string}")
        self._test_basic_inner(ivf, matcher, num_threads,
            skip_thread_test = False,
            test_single_query = test_single_query,
        )

        print("Load and Assemble from clustering")
        clustering=svs.Clustering.load_clustering(test_ivf_clustering)
        ivf = svs.IVF.assemble_from_clustering(
            clustering = clustering,
            data_loader = loader,
            distance = svs.DistanceType.L2,
            num_threads = num_threads
        )
        print(f"Testing: {ivf.experimental_backend_string}")
        self._test_basic_inner(ivf, matcher, num_threads,
            skip_thread_test = False,
            test_single_query = test_single_query,
        )

        # Test saving and reloading.
        print("Testing save and load")
        with TemporaryDirectory() as tempdir:
            configdir = os.path.join(tempdir, "config")
            datadir = os.path.join(tempdir, "data")
            ivf.save(configdir, datadir)

            # Reload from saved directories.
            reloaded = svs.IVF.load(
                config_directory = configdir,
                data_directory = datadir,
                distance = svs.DistanceType.L2,
                num_threads = num_threads
            )

            print(f"Testing reloaded: {reloaded.experimental_backend_string}")
            self._test_basic_inner(
                reloaded,
                matcher,
                num_threads,
                skip_thread_test = True,
            )

    def test_basic(self):
        # Load the index from files.
        default_loader = svs.VectorDataLoader(
            test_data_svs, svs.DataType.float32, dims = test_data_dims
        )
        self._setup(default_loader)

        # Standard tests - run single query test only on first iteration
        is_first = True
        for loader, matcher in self.loader_and_matcher:
            self._test_basic(loader, matcher, test_single_query=is_first)
            is_first = False

    def _groundtruth_map(self):
        return {
            svs.DistanceType.L2: test_groundtruth_l2,
            svs.DistanceType.MIP: test_groundtruth_mip,
            svs.DistanceType.Cosine: test_groundtruth_cosine,
        }

    def _test_build(
        self,
        loader,
        distance: svs.DistanceType,
        matcher
    ):
        num_threads = 2
        distance_map = self._distance_map()

        params = self._get_build_parameters(
            'ivf_test_build', distance_map[distance], matcher
        )

        clustering = svs.Clustering.build(
                build_parameters = params,
                data_loader = loader,
                distance = distance,
                num_threads = num_threads
        )

        ivf = svs.IVF.assemble_from_clustering(
                clustering = clustering,
                data_loader = loader,
                distance = distance,
                num_threads = num_threads,
        )

        print(f"Building: {ivf.experimental_backend_string}")

        groundtruth_map = self._groundtruth_map()
        # Load the queries and groundtruth
        queries = svs.read_vecs(test_queries)
        print(f"Loading groundtruth for: {distance}")
        groundtruth = svs.read_vecs(groundtruth_map[distance])

        # Ensure the number of threads was propagated correctly.
        self.assertEqual(ivf.num_threads, num_threads)

        expected_results = self._get_config_and_recall(
            'ivf_test_build', distance_map[distance], matcher
        )

        for expected in expected_results:
            n_probes, k_reorder, k, nq, expected_recall = \
                self._parse_config_and_recall(expected)

            parameters = svs.IVFSearchParameters(
                n_probes = n_probes,
                k_reorder = k_reorder
            )
            ivf.search_parameters = parameters
            self.assertEqual(ivf.search_parameters.n_probes, n_probes)
            self.assertEqual(ivf.search_parameters.k_reorder, k_reorder)

            results = ivf.search(get_test_set(queries, nq), k)
            recall = svs.k_recall_at(get_test_set(groundtruth, nq), results[0], k, k)
            print(f"Recall = {recall}, Expected = {expected_recall}")
            if not DEBUG:
                self.assertTrue(isapprox(recall, expected_recall, epsilon = 0.005))

    def test_build(self):
        # Build directly from data
        queries = svs.read_vecs(test_queries)

        # Build from file loader
        loader = svs.VectorDataLoader(test_data_svs, svs.DataType.float32)
        matcher = UncompressedMatcher("bfloat16")
        self._test_build(loader, svs.DistanceType.L2, matcher)
        self._test_build(loader, svs.DistanceType.MIP, matcher)
