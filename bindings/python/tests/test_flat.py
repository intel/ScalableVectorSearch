#
# Copyright (C) 2023, Intel Corporation
#
# You can redistribute and/or modify this software under the terms of the
# GNU Affero General Public License version 3.
#
# You should have received a copy of the GNU Affero General Public License
# version 3 along with this software. If not, see
# <https://www.gnu.org/licenses/agpl-3.0.en.html>.
#

# Tests for the Flat index portion of the SVS module.
import unittest
import svs

import numpy as np

# Local dependencies
from .common import \
    isapprox, \
    test_data_svs, \
    test_data_vecs, \
    test_data_dims, \
    test_graph, \
    test_queries, \
    test_groundtruth_l2, \
    test_groundtruth_mip, \
    test_number_of_vectors, \
    test_dimensions

class FlatTester(unittest.TestCase):
    """
    Test index querying.

    NOTE: The structure of these tests closely follows the integration tests in the C++
    library.
    """

    def _loaders(self, file: svs.VectorDataLoader):
        """
        Return a list of loaders to test with exhaustive search.
        Each entry in the list contains a tuple `(loader, recall_dict)` where
        * `loader`: The actual SVS compressed loader object.
        * `recall_dict`: A dictionary mapping distance type to expected recall after the
           exhaustive search.
        """
        return [
            (file, {
                svs.DistanceType.L2: 1.0,
                svs.DistanceType.MIP: 1.0,
            }),
        ]

    def _do_test(self, flat, queries, groundtruth, expected_recall = 1.0):
        """
        Perform a series of tests on a Flat index to test its conformance to expectations.
        Parameters:
            - `flat`: A svs.Flat index manager.
            - `queries`: The set of queries.
            - `groundtruth`: The groundtruth for these queries.

        Tests:
            - Setting of the `batch_size` parameter.
            - Results of `search` are within acceptable margins of the groundtruth.
            - The number of threads can be changed with an observable side-effect.
        """
        # Data interface
        self.assertEqual(flat.size, test_number_of_vectors)
        self.assertEqual(flat.dimensions, test_dimensions)

        # Test setting the batch size
        p = flat.search_parameters
        self.assertEqual(p.data_batch_size, 0)
        self.assertEqual(p.query_batch_size, 0)

        p.data_batch_size = 20
        p.query_batch_size = 10
        self.assertEqual(p.data_batch_size, 20)
        self.assertEqual(p.query_batch_size, 10)

        # Ensure that round-tripping works
        flat.search_parameters = p
        q = flat.search_parameters

        self.assertEqual(q.data_batch_size, 20)
        self.assertEqual(q.query_batch_size, 10)
        q.data_batch_size = 0
        q.query_batch_size = 0
        flat.search_parameters = p

        # Test string formatting.
        self.assertEqual(
            str(q),
            "svs.FlatSearchParameters(data_batch_size = 0, query_batch_size = 0)"
        )

        # Test querying.
        # Return as many neighbors as we have exising groundtruth for.
        num_neighbors = groundtruth.shape[-1]
        results = flat.search(queries, num_neighbors)

        # Compute the recall - should be almost exact.
        # The reason it isn't precisely exact is due to how ties are handled at the very
        # end of the returned neighbor list.
        recall = svs.k_recall_at(groundtruth, results[0], num_neighbors, num_neighbors)
        print(f"Flat. Expected {expected_recall}. Got {recall}.")
        self.assertTrue(isapprox(recall, expected_recall, epsilon = 0.0001))
        # test_threading(flat, queries, num_neighbors)

    def _do_test_from_file(self, distance: svs.DistanceType, queries, groundtruth):
        # Load the index from files.
        num_threads = 2
        loaders = self._loaders(
            svs.VectorDataLoader(
                test_data_svs, svs.DataType.float32, dims = test_data_dims
            )
        );
        for loader, recall in loaders:
            index = svs.Flat(
                loader,
                distance = distance,
                num_threads = num_threads
            )

            self.assertEqual(index.num_threads, num_threads)
            self._do_test(index, queries, groundtruth, expected_recall = recall[distance])

    def test_from_file(self):
        """
        Test basic querying.
        """
        queries = svs.read_vecs(test_queries)
        # Euclidean Distance
        self._do_test_from_file(
            svs.DistanceType.L2,
            queries,
            svs.read_vecs(test_groundtruth_l2)
        )
        # Inner Product
        self._do_test_from_file(
            svs.DistanceType.MIP,
            queries,
            svs.read_vecs(test_groundtruth_mip)
        )

    def test_from_array(self):
        data_f32 = svs.read_vecs(test_data_vecs)
        queries_f32 = svs.read_vecs(test_queries)
        groundtruth = svs.read_vecs(test_groundtruth_l2)

        # Test `float32`
        print("Flat, From Array, Float32")
        flat = svs.Flat(data_f32, svs.DistanceType.L2)
        self._do_test(flat, queries_f32, groundtruth)

        # Test `float16`
        print("Flat, From Array, Float16")
        data_f16 = data_f32.astype('float16')
        queries_f16 = queries_f32.astype('float16')
        flat = svs.Flat(data_f16, svs.DistanceType.L2)
        self._do_test(flat, queries_f16, groundtruth)

        # Test `int8`
        print("Flat, From Array, Int8")
        data_i8 = data_f32.astype('int8')
        queries_i8 = queries_f32.astype('int8')
        flat = svs.Flat(data_i8, svs.DistanceType.L2)
        self._do_test(flat, queries_i8, groundtruth)

        # Test 'uint8'
        # The dataset is stored as values that can be encoded as `int8`.
        # To test `uint8`, we need to apply a shift by 128 to make all values losslessly
        # encodable as `uint8` types.
        print("Flat, From Array, UInt8")
        data_u8 = (data_f32 + 128).astype('uint8')
        queries_u8 = (queries_f32 + 128).astype('uint8')
        flat = svs.Flat(data_u8, svs.DistanceType.L2)
        self._do_test(flat, queries_u8, groundtruth)
