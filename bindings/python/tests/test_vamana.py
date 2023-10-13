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

# Tests for the Vamana index portion of the PySVS module.
import unittest
import os
import warnings

from tempfile import TemporaryDirectory

import pysvs

# Local dependencies
from .common import \
    isapprox, \
    test_data_svs, \
    test_data_vecs, \
    test_data_dims, \
    test_graph, \
    test_vamana_config, \
    test_queries, \
    test_groundtruth_l2, \
    test_groundtruth_mip, \
    test_groundtruth_cosine, \
    test_number_of_vectors, \
    test_dimensions, \
    timed

DEBUG = False;

class VamanaTester(unittest.TestCase):
    """
    Test index querying, building, and saving.

    NOTE: The structure of these tests closely follows the integration tests in the C++
    library.
    """
    def setUp(self):
        # Initialize expected recall tables.
        # This should be kept in-sync with the C++ unit tests.
        #
        # Keys:
        # - Both the search window size to use and the number of neighbors to return.
        # Values:
        # - The expected recall for the given search-window-size and number of neighbors.
        self.recall_l2 = {
            2: 0.4595,
            3: 0.537333,
            4: 0.60025,
            5: 0.643,
            10: 0.7585,
            20: 0.86,
            50: 0.94662,
            100: 0.97724,
        }

    def _setup(self, loader: pysvs.VectorDataLoader):
        self.loader_and_recall = [
            (loader, self.recall_l2),
            (pysvs.LVQ8(loader, 0), {
                2: 0.4575, 3: 0.53833, 4: 0.59974, 5: 0.6438, 10: 0.7584,
            }),
            (pysvs.LVQ4x4(loader, 0), {
                2: 0.4225, 3: 0.498, 4: 0.55825, 5: 0.5966, 10: 0.7055,
            }),
            (pysvs.LVQ4x8(loader, 0), {
                2: 0.4225, 3: 0.498, 4: 0.55825, 5: 0.5966, 10: 0.7055,
            }),
            (pysvs.LVQ8x8(loader, 0), {
                2: 0.4575, 3: 0.538333, 4: 0.59975, 5: 0.6438, 10: 0.7584,
            }),
        ]

    def _test_basic_inner(
            self,
            vamana: pysvs.Vamana,
            recall_dict,
            num_threads: int,
            skip_thread_test: bool = False,
        ):
        # Make sure that the number of threads is propagated correctly.
        self.assertEqual(vamana.num_threads, num_threads)
        self.assertFalse(vamana.visited_set_enabled)

        # load the queries and groundtruth
        queries = pysvs.read_vecs(test_queries)
        groundtruth = pysvs.read_vecs(test_groundtruth_l2)

        self.assertEqual(queries.shape, (1000, 128))
        self.assertEqual(groundtruth.shape, (1000, 100))

        # Data interface
        self.assertEqual(vamana.size, test_number_of_vectors)
        self.assertEqual(vamana.dimensions, test_dimensions)

        # Test setting the window size.
        vamana.search_window_size = 20
        self.assertEqual(vamana.search_window_size, 20)

        vamana.search_window_size = 10
        self.assertEqual(vamana.search_window_size, 10)

        for (search_window_size, expected_recall) in recall_dict.items():
            for visited_set_enabled in (True, False):
                vamana.visited_set_enabled = visited_set_enabled
                self.assertEqual(vamana.visited_set_enabled, visited_set_enabled)

                # Set the search window size and verify that the change to the search window
                # size variable is visible.
                vamana.search_window_size = search_window_size
                self.assertEqual(vamana.search_window_size, search_window_size)
                results = vamana.search(queries, search_window_size)

                recall = pysvs.k_recall_at(
                    groundtruth,
                    results[0],
                    search_window_size,
                    search_window_size
                )
                print(f"Recall = {recall}, Expected = {expected_recall}")
                if not DEBUG:
                    self.assertTrue(isapprox(recall, expected_recall, epsilon = 0.0005))

        # Disable visited set.
        self.visited_set_enabled = False

        if skip_thread_test:
            return

        # # Makes sure that setting the number of threads works correctly.
        # for (search_window_size, expected_recall) in self.recall_l2.items():
        #     # Abort for larger search window sizes to keep overall runtime down.
        #     if search_window_size > 40:
        #         continue

        #     vamana.search_window_size = search_window_size
        #     test_threading(vamana, queries, search_window_size, iters = 10)


    def _test_basic(self, loader, recall_dict):
        num_threads = 2
        vamana = pysvs.Vamana(
            test_vamana_config,
            pysvs.GraphLoader(test_graph),
            loader,
            pysvs.DistanceType.L2,
            num_threads = num_threads
        )

        self._test_basic_inner(vamana, recall_dict, num_threads)

        # Test saving and reloading.
        with TemporaryDirectory() as tempdir:
            configdir = os.path.join(tempdir, "config")
            graphdir = os.path.join(tempdir, "graph")
            datadir = os.path.join(tempdir, "data")
            vamana.save(configdir, graphdir, datadir);

            # Figure out how to reload the index.
            if isinstance(loader, pysvs.VectorDataLoader):
                reloader = type(loader)(datadir, pysvs.DataType.float32)
            else:
                reloader = type(loader)(datadir, test_data_dims)

            reloaded = pysvs.Vamana(
                configdir,
                pysvs.GraphLoader(graphdir),
                reloader,
                pysvs.DistanceType.L2,
            )

            reloaded.num_threads = num_threads
            self._test_basic_inner(
                reloaded, recall_dict, num_threads, skip_thread_test = True
            )

    def test_basic(self):
        # Load the index from files.
        self._setup(
            pysvs.VectorDataLoader(
                test_data_svs, pysvs.DataType.float32, dims = test_data_dims
            )
        )
        for loader, recall_dict in self.loader_and_recall:
            self._test_basic(loader, recall_dict)

    def test_deprecation(self):
        with warnings.catch_warnings(record = True) as w:
            p = pysvs.VamanaBuildParameters(num_threads = 1)
            self.assertTrue(len(w) == 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertTrue("VamanaBuildParameters" in str(w[0].message))

    def _alpha_map(self):
        return {
            pysvs.DistanceType.L2: 1.2,
            pysvs.DistanceType.MIP: 1.0 / 1.2,
            pysvs.DistanceType.Cosine: 1.0 / 1.2,
        }

    def _groundtruth_map(self):
        return {
            pysvs.DistanceType.L2: test_groundtruth_l2,
            pysvs.DistanceType.MIP: test_groundtruth_mip,
            pysvs.DistanceType.Cosine: test_groundtruth_cosine,
        }

    def _test_build(self, data, distance: pysvs.DistanceType):
        # Map from the distance to the distance type to use
        alpha_map = self._alpha_map()
        groundtruth_map = self._groundtruth_map()

        result_map = {
            # Expected results for the L2 Distance
            pysvs.DistanceType.L2 : {
                2: 0.2185,
                3: 0.264333,
                4: 0.3045,
                5: 0.3308,
                10: 0.4383,
                20: 0.5398,
                50: 0.66378,
                100: 0.74339,
            },
            # Expected results for the MIP Distance
            pysvs.DistanceType.MIP : {
                2: 0.09,
                3: 0.11566667,
                4: 0.143,
                5: 0.1642,
                10: 0.242,
                20: 0.35485,
                50: 0.53504,
                100: 0.68658,
            },
            # Expected results for Cosine Distance
            pysvs.DistanceType.Cosine : {
                2: 0.0785,
                3: 0.0976666667,
                4: 0.1165,
                5: 0.1392,
                10: 0.2136,
                20: 0.32545,
                50: 0.51474,
                100: 0.67853,
            },
        }

        num_threads = 2
        parameters = pysvs.VamanaBuildParameters(
            alpha = alpha_map[distance],
            graph_max_degree = 30,
            window_size = 40,
            max_candidate_pool_size = 30,
        )
        vamana = pysvs.Vamana.build(parameters, data, distance, num_threads = num_threads)

        # Load the queries and groundtruth
        queries = pysvs.read_vecs(test_queries)
        print(f"Loading groundtruth for: {distance}")
        groundtruth = pysvs.read_vecs(groundtruth_map[distance])

        # Ensure the number of threads was propagated correctly.
        self.assertEqual(vamana.num_threads, 2)

        # Run through the search window sizes and make sure we get similar results to the
        # C++ tests.
        for (search_window_size, expected_recall) in result_map[distance].items():
            vamana.search_window_size = search_window_size
            self.assertEqual(vamana.search_window_size, search_window_size)
            results = vamana.search(queries, search_window_size)
            recall = pysvs.k_recall_at(
                groundtruth,
                results[0],
                search_window_size,
                search_window_size
            )
            print(f"Search Window: {search_window_size}, Got recall {recall}. Expected {expected_recall}")
            self.assertTrue(isapprox(recall, expected_recall, epsilon = 0.001))

    def test_build(self):
        # Build directly from data
        data = pysvs.read_vecs(test_data_vecs)
        self._test_build(data, pysvs.DistanceType.L2)
        self._test_build(data, pysvs.DistanceType.MIP)
        self._test_build(data, pysvs.DistanceType.Cosine)

        # Build using float16
        data_f16 = data.astype('float16')
        self._test_build(data_f16, pysvs.DistanceType.L2)
        self._test_build(data_f16, pysvs.DistanceType.MIP)
        self._test_build(data_f16, pysvs.DistanceType.Cosine)

        # Build from file loader
        loader = pysvs.VectorDataLoader(test_data_svs, pysvs.DataType.float32)
        self._test_build(loader, pysvs.DistanceType.L2)
        self._test_build(loader, pysvs.DistanceType.MIP)
        self._test_build(loader, pysvs.DistanceType.Cosine)

    def _test_build_quantized(self, compressor, distance: pysvs.DistanceType):
        alpha_map = self._alpha_map()
        groundtruth_map = self._groundtruth_map()

        result_map = {
            # Expected results for the L2 Distance
            pysvs.DistanceType.L2 : {
                2: 0.213,
                3: 0.2626667,
                4: 0.30675,
                5: 0.3312,
                10: 0.4357,
                20: 0.54075,
                50: 0.66342,
                100: 0.74416,
            },
            # Expected results for the MIP Distance
            pysvs.DistanceType.MIP : {
                2: 0.098,
                3: 0.124,
                4: 0.14725,
                5: 0.1648,
                10: 0.2512,
                20: 0.3574,
                50: 0.53772,
                100: 0.68581,
            },
        }

        num_threads = 2
        parameters = pysvs.VamanaBuildParameters(
            alpha = alpha_map[distance],
            graph_max_degree = 30,
            window_size = 40,
            max_candidate_pool_size = 30,
        )
        vamana = pysvs.Vamana.build(
            parameters, compressor, distance, num_threads = num_threads
        )

        # Load the queries and groundtruth
        queries = pysvs.read_vecs(test_queries)
        groundtruth = pysvs.read_vecs(groundtruth_map[distance])

        # Ensure the number of threads was propagated correctly.
        self.assertEqual(vamana.num_threads, 2)

        # Run through the search window sizes and make sure we get similar results to the
        # C++ tests.
        print(f"Running tests for compressor {compressor}")
        for (search_window_size, expected_recall) in result_map[distance].items():
            vamana.search_window_size = search_window_size
            self.assertEqual(vamana.search_window_size, search_window_size)
            results = vamana.search(queries, search_window_size)
            recall = pysvs.k_recall_at(
                groundtruth,
                results[0],
                search_window_size,
                search_window_size
            )
            print(f"Search Window: {search_window_size}, Got recall {recall}. Expected {expected_recall}")
            self.assertTrue(isapprox(recall, expected_recall, epsilon = 0.001))

    def test_quantized_build(self):
        # Build from file loader
        loader = pysvs.VectorDataLoader(test_data_svs, pysvs.DataType.float32, dims = 128)
        compressor = pysvs.LVQ8(loader)

        self._test_build_quantized(compressor, pysvs.DistanceType.L2)
        self._test_build_quantized(compressor, pysvs.DistanceType.MIP)
