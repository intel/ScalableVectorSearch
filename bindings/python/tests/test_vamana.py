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
import toml

import numpy as np

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
    test_vamana_reference, \
    test_number_of_vectors, \
    test_dimensions, \
    timed, \
    get_test_set

from .dataset import \
    UncompressedMatcher, \
    LVQMatcher, \
    LeanVecMatcher

DEBUG = False;

class VamanaTester(unittest.TestCase):
    """
    Test index querying, building, and saving.

    NOTE: The structure of these tests closely follows the integration tests in the C++
    library. Configurations and recalls values are used from the common reference file created
    using the benchmarking infrastructure
    """
    def setUp(self):
        # Initialize expected results from the common reference file
        with open(test_vamana_reference) as f:
            self.reference_results = toml.load(f)

    def _setup(self, loader: pysvs.VectorDataLoader):
        sequential = pysvs.LVQStrategy.Sequential
        turbo = pysvs.LVQStrategy.Turbo

        # Generate LeanVec OOD matrices
        data = pysvs.read_vecs(test_data_vecs)
        queries = pysvs.read_vecs(test_queries)
        data_matrix, query_matrix = pysvs.compute_leanvec_matrices(data, queries, 64);

        self.loader_and_matcher = [
            (loader, UncompressedMatcher("float32")),
            # LVQ
            (pysvs.LVQLoader(loader, primary = 8, padding = 0), LVQMatcher(8)),
            (pysvs.LVQLoader(loader, primary = 4, padding = 0), LVQMatcher(4)),
            (pysvs.LVQLoader(
                loader, primary = 4, residual = 8, strategy = sequential, padding = 0),
                LVQMatcher(4, 8)
            ),
            (pysvs.LVQLoader(
                loader, primary = 4, residual = 8, strategy = turbo, padding = 0),
                LVQMatcher(4, 8)
            ),
            (pysvs.LVQLoader(
                loader, primary = 8, residual = 8, padding = 0),
                LVQMatcher(8, 8)
            ),

            #LeanVec
            (
                pysvs.LeanVecLoader(
                    loader,
                    leanvec_dims = 64,
                    primary_kind = pysvs.LeanVecKind.float32,
                    secondary_kind = pysvs.LeanVecKind.float32,
                ),
                LeanVecMatcher("float32", "float32", 64)
            ),
            # (
            #     pysvs.LeanVecLoader(
            #         loader,
            #         leanvec_dims = 64,
            #         primary_kind = pysvs.LeanVecKind.lvq4,
            #         secondary_kind = pysvs.LeanVecKind.lvq4,
            #     ),
            #     LeanVecMatcher("lvq4", "lvq4", 64)
            # ),
            # (
            #     pysvs.LeanVecLoader(
            #         loader,
            #         leanvec_dims = 64,
            #         primary_kind = pysvs.LeanVecKind.lvq4,
            #         secondary_kind = pysvs.LeanVecKind.lvq8,
            #     ),
            #     LeanVecMatcher("lvq4", "lvq8", 64),
            # ),
            # (
            #     pysvs.LeanVecLoader(
            #         loader,
            #         leanvec_dims = 64,
            #         primary_kind = pysvs.LeanVecKind.lvq8,
            #         secondary_kind = pysvs.LeanVecKind.lvq4,
            #     ),
            #     LeanVecMatcher("lvq8", "lvq4", 64)
            # ),
            (
                pysvs.LeanVecLoader(
                    loader,
                    leanvec_dims = 64,
                    primary_kind = pysvs.LeanVecKind.lvq8,
                    secondary_kind = pysvs.LeanVecKind.lvq8,
                    alignment = 0
                ),
                LeanVecMatcher("lvq8", "lvq8", 64)
            ),
            (
                pysvs.LeanVecLoader(
                    loader,
                    leanvec_dims = 96,
                    primary_kind = pysvs.LeanVecKind.float32,
                    secondary_kind = pysvs.LeanVecKind.float32,
                    alignment = 0
                ),
                LeanVecMatcher("float32", "float32", 96)
            ),

            # LeanVec OOD
            (
                pysvs.LeanVecLoader(
                    loader,
                    leanvec_dims = 64,
                    primary_kind = pysvs.LeanVecKind.float32,
                    secondary_kind = pysvs.LeanVecKind.float32,
                    data_matrix = data_matrix,
                    query_matrix = query_matrix,
                    alignment = 0
                ),
                LeanVecMatcher("float32", "float32", 64, False)
            ),
            (
                pysvs.LeanVecLoader(
                    loader,
                    leanvec_dims = 64,
                    primary_kind = pysvs.LeanVecKind.lvq8,
                    secondary_kind = pysvs.LeanVecKind.lvq8,
                    data_matrix = data_matrix,
                    query_matrix = query_matrix,
                    alignment = 0
                ),
                LeanVecMatcher("lvq8", "lvq8", 64, False)
            )
        ]

    def _distance_map(self):
        return {
            pysvs.DistanceType.L2: "L2",
            pysvs.DistanceType.MIP: "MIP",
            pysvs.DistanceType.Cosine: "Cosine",
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
        size = params['search_window_size']
        capacity = params['search_buffer_capacity']
        k = results['num_neighbors']
        nq  = results['num_queries']
        recall = results['recall']
        return size, capacity, k, nq, recall

    def _get_build_parameters(self, test_type, distance, matcher):
        params = []
        for results in self.reference_results[test_type]:
            if (results['distance'] == distance) and matcher.is_match(results['dataset']):
                params.append(results['build_parameters'])

        assert len(params) == 1, "Should match one parameters entry!"
        params = params[0]

        return pysvs.VamanaBuildParameters(
            alpha = params["alpha"],
            graph_max_degree = params["graph_max_degree"],
            prune_to = params["prune_to"],
            window_size = params["window_size"],
            max_candidate_pool_size = params["max_candidate_pool_size"]
        )

    # Ensure that passing 1-dimensional queries works and produces the same results as
    # query batches.
    def _test_single_query(
            self,
            vamana: pysvs.Vamana,
            queries
        ):

        I_full, D_full = vamana.search(queries, 10);

        I_single = []
        D_single = []
        for i in range(queries.shape[0]):
            query = queries[i, :]
            self.assertTrue(query.ndim == 1)
            I, D = vamana.search(query, 10)

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
            vamana.search(queries_3d, 10)

        self.assertTrue("only accept numpy vectors or matrices" in str(context.exception))

    def _test_basic_inner(
            self,
            vamana: pysvs.Vamana,
            matcher,
            num_threads: int,
            skip_thread_test: bool = False,
            first_iter: bool = False,
            test_single_query: bool = False,
        ):
        # Make sure that the number of threads is propagated correctly.
        self.assertEqual(vamana.num_threads, num_threads)

        # load the queries and groundtruth
        queries = pysvs.read_vecs(test_queries)
        groundtruth = pysvs.read_vecs(test_groundtruth_l2)

        self.assertEqual(queries.shape, (1000, 128))
        self.assertEqual(groundtruth.shape, (1000, 100))

        # Data interface
        self.assertEqual(vamana.size, test_number_of_vectors)

        # The dimensionality exposed by the index should always match the original
        # dataset dimensions.
        self.assertEqual(vamana.dimensions, test_dimensions)

        # Test setting the window size.
        vamana.search_window_size = 20
        self.assertEqual(vamana.search_window_size, 20)

        vamana.search_window_size = 10
        self.assertEqual(vamana.search_window_size, 10)

        expected_results = self._get_config_and_recall('vamana_test_search', 'L2', matcher)
        for expected in expected_results:
            window_size, buffer_capacity, k, nq, expected_recall = \
                self._parse_config_and_recall(expected)

            for visited_set_enabled in (True, False):
                parameters = pysvs.VamanaSearchParameters(
                    pysvs.SearchBufferConfig(window_size, buffer_capacity),
                    visited_set_enabled
                )
                vamana.search_parameters = parameters
                self.assertEqual(vamana.search_parameters, parameters)

                results = vamana.search(get_test_set(queries, nq), k)
                recall = pysvs.k_recall_at(get_test_set(groundtruth, nq), results[0], k, k)
                print(f"Recall = {recall}, Expected = {expected_recall}")
                if not DEBUG:
                    self.assertTrue(isapprox(recall, expected_recall, epsilon = 0.0005))

        if test_single_query:
            self._test_single_query(vamana, queries)

        # Test calibration if this is the first iteration.
        if not first_iter:
            return

        # Perform calibration with the first result
        window_size, buffer_capacity, k, nq, target_recall = \
            self._parse_config_and_recall(expected_results[0])

        p = vamana.experimental_calibrate(
            get_test_set(queries, nq), get_test_set(groundtruth, nq), k, target_recall
        )
        I, _ = vamana.search(get_test_set(queries, nq), k)
        recall = pysvs.k_recall_at(get_test_set(groundtruth, nq), I, k, k)
        self.assertTrue(recall >= target_recall)

        # Ensure that disabling prefetch tuning does not mutate the result
        p.prefetch_lookahead = 0
        p.prefetch_step = 0
        vamana.search_parameters = p

        calibration_parameters = pysvs.VamanaCalibrationParameters()
        calibration_parameters.train_prefetchers = False
        q = vamana.experimental_calibrate(
            get_test_set(queries, nq), get_test_set(groundtruth, nq), \
            k, target_recall, calibration_parameters
        )
        self.assertTrue(recall >= target_recall)
        self.assertEqual(q.prefetch_lookahead, 0)
        self.assertEqual(q.prefetch_step, 0)

    def _test_basic(self, loader, matcher, first_iter: bool = False):
        num_threads = 2
        vamana = pysvs.Vamana(
            test_vamana_config,
            pysvs.GraphLoader(test_graph),
            loader,
            pysvs.DistanceType.L2,
            num_threads = num_threads
        )

        print(f"Testing: {vamana.experimental_backend_string}")
        self._test_basic_inner(vamana, matcher, num_threads,
            skip_thread_test = False,
            first_iter = first_iter,
            test_single_query = first_iter,
        )

        # Test saving and reloading.
        with TemporaryDirectory() as tempdir:
            configdir = os.path.join(tempdir, "config")
            graphdir = os.path.join(tempdir, "graph")
            datadir = os.path.join(tempdir, "data")
            vamana.save(configdir, graphdir, datadir);

            # Reload from raw-files.
            reloaded = pysvs.Vamana(configdir, graphdir, datadir, pysvs.DistanceType.L2)

            # Backend strings should match unless this is LVQ loader with a Turbo backend
            # TODO: Allow for more introspection in the LVQLoader fields.
            if not isinstance(loader, pysvs.LVQLoader):
                self.assertTrue(
                    vamana.experimental_backend_string ==
                    reloaded.experimental_backend_string
                )

            reloaded.num_threads = num_threads
            self._test_basic_inner(
                reloaded,
                matcher,
                num_threads,
                skip_thread_test = True,
                first_iter = first_iter,
            )

    def test_basic(self):
        # Load the index from files.
        default_loader = pysvs.VectorDataLoader(
            test_data_svs, pysvs.DataType.float32, dims = test_data_dims
        )
        self._setup(default_loader)

        # Standard tests
        first_iter = True
        for loader, matcher in self.loader_and_matcher:
            self._test_basic(loader, matcher, first_iter = first_iter)
            first_iter = False

    def test_lvq_reload(self):
        # Test LVQ reloading with different alignemnts and strategies.
        default_loader = pysvs.VectorDataLoader(
            test_data_svs, pysvs.DataType.float32, dims = test_data_dims
        )

        lvq_loader = pysvs.LVQLoader(
            default_loader,
            primary = 4,
            residual = 8,
            strategy = pysvs.LVQStrategy.Sequential
        );
        matcher = LVQMatcher(4, 8)

        num_threads = 2
        vamana = pysvs.Vamana(
            test_vamana_config,
            pysvs.GraphLoader(test_graph),
            lvq_loader,
            pysvs.DistanceType.L2,
            num_threads = num_threads
        )

        print(f"Testing: {vamana.experimental_backend_string}")
        self._test_basic_inner(
            vamana,
            matcher,
            num_threads,
            skip_thread_test = False,
            first_iter = False,
        )

        # Test saving and reloading.
        with TemporaryDirectory() as tempdir:
            configdir = os.path.join(tempdir, "config")
            graphdir = os.path.join(tempdir, "graph")
            datadir = os.path.join(tempdir, "data")
            vamana.save(configdir, graphdir, datadir)

            reloader = pysvs.LVQLoader(
                datadir,
                strategy = pysvs.LVQStrategy.Sequential,
                padding = 32,
            )

            print("Reloading LVQ with padding")
            self._test_basic_inner(
                pysvs.Vamana(configdir, graphdir, reloader, num_threads = num_threads),
                matcher,
                num_threads,
                skip_thread_test = False,
                first_iter = False,
            )

            reloader = pysvs.LVQLoader(
                datadir, strategy = pysvs.LVQStrategy.Turbo, padding = 32,
            )

            print("Reloading LVQ as Turbo")
            self._test_basic_inner(
                pysvs.Vamana(configdir, graphdir, reloader, num_threads = num_threads),
                matcher,
                num_threads,
                skip_thread_test = False,
                first_iter = False,
            )

    def test_deprecation(self):
        with warnings.catch_warnings(record = True) as w:
            p = pysvs.VamanaBuildParameters(num_threads = 1)
            self.assertTrue(len(w) == 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertTrue("VamanaBuildParameters" in str(w[0].message))

    def _groundtruth_map(self):
        return {
            pysvs.DistanceType.L2: test_groundtruth_l2,
            pysvs.DistanceType.MIP: test_groundtruth_mip,
            pysvs.DistanceType.Cosine: test_groundtruth_cosine,
        }

    def _test_build(
        self,
        loader,
        distance: pysvs.DistanceType,
        matcher,
        additional_query_types = []
    ):
        num_threads = 2
        distance_map = self._distance_map()

        params = self._get_build_parameters(
            'vamana_test_build', distance_map[distance], matcher
        );

        vamana = pysvs.Vamana.build(params, loader, distance, num_threads = num_threads)
        print(f"Building: {vamana.experimental_backend_string}")

        groundtruth_map = self._groundtruth_map()
        # Load the queries and groundtruth
        queries = pysvs.read_vecs(test_queries)
        print(f"Loading groundtruth for: {distance}")
        groundtruth = pysvs.read_vecs(groundtruth_map[distance])

        # Ensure the number of threads was propagated correctly.
        self.assertEqual(vamana.num_threads, num_threads)

        expected_results = self._get_config_and_recall(
            'vamana_test_build', distance_map[distance], matcher
        )

        for expected in expected_results:
            window_size, buffer_capacity, k, nq, expected_recall = \
                self._parse_config_and_recall(expected)

            parameters = pysvs.VamanaSearchParameters(
                pysvs.SearchBufferConfig(window_size, buffer_capacity), False
            )
            vamana.search_parameters = parameters
            self.assertEqual(vamana.search_parameters, parameters)

            results = vamana.search(get_test_set(queries, nq), k)
            recall = pysvs.k_recall_at(get_test_set(groundtruth, nq), results[0], k, k)
            print(f"Recall = {recall}, Expected = {expected_recall}")
            if not DEBUG:
                self.assertTrue(isapprox(recall, expected_recall, epsilon = 0.005))

            for typ in additional_query_types:
                print(f"Trying Query Type {typ}")
                self.assertTrue(pysvs.np_to_svs(typ) in vamana.query_types)
                queries_converted = queries.astype(typ)
                results = vamana.search(get_test_set(queries_converted, nq), k)
                recall = pysvs.k_recall_at(get_test_set(groundtruth, nq), results[0], k, k)
                print(f"Recall = {recall}, Expected = {expected_recall}")

                if not DEBUG:
                    self.assertTrue(isapprox(recall, expected_recall, epsilon = 0.005))

    def test_build(self):
        # Build directly from data
        data = pysvs.read_vecs(test_data_vecs)

        # Generate LeanVec OOD matrices
        queries = pysvs.read_vecs(test_queries)
        data_matrix, query_matrix = pysvs.compute_leanvec_matrices(data, queries, 64);

        matcher = UncompressedMatcher("float32")
        self._test_build(data, pysvs.DistanceType.L2, matcher)
        self._test_build(data, pysvs.DistanceType.MIP, matcher)
        self._test_build(data, pysvs.DistanceType.Cosine, matcher)

        # Build using float16
        data_f16 = data.astype('float16')
        matcher = UncompressedMatcher("float16")
        f16 = [np.float16]
        self._test_build(
            data_f16, pysvs.DistanceType.L2, matcher, additional_query_types = f16
        )
        self._test_build(
            data_f16, pysvs.DistanceType.MIP, matcher, additional_query_types = f16
        )
        self._test_build(
            data_f16, pysvs.DistanceType.Cosine, matcher, additional_query_types = f16
        )

        # Build from file loader
        loader = pysvs.VectorDataLoader(test_data_svs, pysvs.DataType.float32)
        matcher = UncompressedMatcher("float32")
        self._test_build(loader, pysvs.DistanceType.L2, matcher)
        self._test_build(loader, pysvs.DistanceType.MIP, matcher)
        self._test_build(loader, pysvs.DistanceType.Cosine, matcher)

        data = pysvs.VectorDataLoader(test_data_svs, pysvs.DataType.float32, dims = 128)

        # Build from LVQ
        loader = pysvs.LVQ8(data)
        matcher = LVQMatcher(8)
        self._test_build(loader, pysvs.DistanceType.L2, matcher)
        self._test_build(loader, pysvs.DistanceType.MIP, matcher)

        loader = pysvs.LVQ4x8(data)
        matcher = LVQMatcher(4, 8)
        self._test_build(loader, pysvs.DistanceType.L2, matcher)
        self._test_build(loader, pysvs.DistanceType.MIP, matcher)

        # Build from LeanVec
        loader = pysvs.LeanVecLoader(
            data,
            leanvec_dims = 64,
            primary_kind = pysvs.LeanVecKind.float32,
            secondary_kind = pysvs.LeanVecKind.float32
        )
        matcher = LeanVecMatcher("float32", "float32", 64)
        self._test_build(loader, pysvs.DistanceType.L2, matcher)
        self._test_build(loader, pysvs.DistanceType.MIP, matcher)

        loader = pysvs.LeanVecLoader(
            data,
            leanvec_dims = 64,
            primary_kind = pysvs.LeanVecKind.lvq8,
            secondary_kind = pysvs.LeanVecKind.lvq8
        )
        matcher = LeanVecMatcher("lvq8", "lvq8", 64)
        self._test_build(loader, pysvs.DistanceType.L2, matcher)
        self._test_build(loader, pysvs.DistanceType.MIP, matcher)

        # Build from LeanVec OOD
        loader = pysvs.LeanVecLoader(
            data,
            leanvec_dims = 64,
            primary_kind = pysvs.LeanVecKind.lvq8,
            secondary_kind = pysvs.LeanVecKind.lvq8,
            data_matrix = data_matrix,
            query_matrix = query_matrix
        )
        matcher = LeanVecMatcher("lvq8", "lvq8", 64, False)
        self._test_build(loader, pysvs.DistanceType.L2, matcher)
        self._test_build(loader, pysvs.DistanceType.MIP, matcher)
