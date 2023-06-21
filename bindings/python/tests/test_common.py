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

# Tests for the exported methods in the module `pysvs.common`.
import tempfile
import unittest
import os
import pysvs

import numpy as np

from .common import \
    isapprox, \
    test_data_vecs, \
    test_data_dims, \
    test_queries, \
    test_groundtruth_l2

class CommonTester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(CommonTester, self).__init__(*args, **kwargs)

        self.tempdir = None
        self.tempdir_name = None

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.tempdir_name = self.tempdir.name

    def tearDown(self):
        if self.tempdir is not None:
            self.tempdir.cleanup()
            self.tempdir = None
            self.tempdir_name = None

    #####
    ##### Tests
    #####

    def test_version(self):
        self.assertEqual(pysvs.library_version(), "v0.0.1")

    def test_random_dataset(self):
        for dtype in (np.float32, np.uint32, np.uint8):
            x = pysvs.common.random_dataset(10000, 10, dtype = dtype, seed = 1234)
            self.assertEqual(x.ndim, 2)
            self.assertEqual(x.shape[0], 10000)
            self.assertEqual(x.shape[1], 10)
            self.assertEqual(x.dtype, dtype)

            # Generate again with the same seed. Ensure the seed is propagated correctly.
            y = pysvs.common.random_dataset(10000, 10, dtype = dtype, seed = 1234)
            self.assertTrue(np.array_equal(x, y))

            # Ensure different seed yields different results.
            z = pysvs.common.random_dataset(10000, 10, dtype = dtype, seed = 5678)
            self.assertFalse(np.array_equal(x, z))

        # Deterministic random number generation.
        # This is a canary test to hopefully catch if deterministic, platform independent
        # random number generation fails.
        #
        # Hopefully `RandomState` does what is says.
        x = pysvs.common.random_dataset(1, 10, dtype = np.float32, seed = 44)
        reference = np.array([[
            -0.7506147, 1.3163574, 1.24614, -1.6049157, -1.4681437,
            -1.7150705, 1.8587837, 0.087587975, -0.052322198, 0.55547166]],
            dtype = np.float32
        )
        self.assertTrue(np.array_equal(x, reference))

    def test_readwrite_vecs(self):
        svs_file = os.path.join(self.tempdir_name, "test.svs")

        # fvecs
        file = os.path.join(self.tempdir_name, "test.fvecs")
        x = pysvs.common.random_dataset(10000, 10, dtype = np.float32)
        pysvs.write_vecs(x, file)
        y = pysvs.read_vecs(file)
        self.assertEqual(y.dtype, np.float32)
        self.assertTrue(np.array_equal(x, y))
        # convert to svs
        pysvs.convert_vecs_to_svs(file, svs_file, dtype = pysvs.float32)
        # z = pysvs.read_svs(svs_file, dtype = np.float32)
        # self.assertEqual(z.dtype, np.float32)
        # self.assertTrue(np.array_equal(x, z))

        # ivecs
        file = os.path.join(self.tempdir_name, "test.ivecs")
        x = pysvs.common.random_dataset(10000, 10, dtype = np.uint32)
        pysvs.write_vecs(x, file)
        y = pysvs.read_vecs(file)
        self.assertEqual(y.dtype, np.uint32)
        self.assertTrue(np.array_equal(x, y))
        # convert to svs
        pysvs.convert_vecs_to_svs(file, svs_file, dtype = pysvs.uint32)
        # z = pysvs.read_svs(svs_file, dtype = np.uint32)
        # self.assertEqual(z.dtype, np.uint32)
        # self.assertTrue(np.array_equal(x, z))

        # bvecs
        file = os.path.join(self.tempdir_name, "test.bvecs")
        x = pysvs.common.random_dataset(10000, 10, dtype = np.uint8)
        pysvs.write_vecs(x, file)
        # y = pysvs.read_vecs(file)
        # self.assertEqual(y.dtype, np.uint8)
        # self.assertTrue(np.array_equal(x, y))
        # convert to svs
        pysvs.convert_vecs_to_svs(file, svs_file, dtype = pysvs.uint8)
        # z = pysvs.read_svs(svs_file, dtype = np.uint8)
        # self.assertEqual(z.dtype, np.uint8)
        # self.assertTrue(np.array_equal(x, z))

    def test_vecs_extension_checking(self):
        # Float
        x = pysvs.common.random_dataset(10, 128, dtype = np.float32)
        self.assertTrue(x.dtype == np.float32)
        self.assertRaises(
            RuntimeError, pysvs.write_vecs, x, os.path.join(self.tempdir_name, "temp.hvecs")
        );

        # Half
        x = pysvs.common.random_dataset(10, 128, dtype = np.float16)
        self.assertTrue(x.dtype == np.float16)
        self.assertRaises(
            RuntimeError, pysvs.write_vecs, x, os.path.join(self.tempdir_name, "temp.fvecs")
        );

        # UInt32
        x = pysvs.common.random_dataset(10, 128, dtype = np.uint32)
        self.assertTrue(x.dtype == np.uint32)
        self.assertRaises(
            RuntimeError, pysvs.write_vecs, x, os.path.join(self.tempdir_name, "temp.bvecs")
        );

        # UInt8
        x = pysvs.common.random_dataset(10, 128, dtype = np.uint8)
        self.assertTrue(x.dtype == np.uint8)
        self.assertRaises(
            RuntimeError, pysvs.write_vecs, x, os.path.join(self.tempdir_name, "temp.ivecs")
        );

    def test_generate_test_dataset(self):
        pysvs.generate_test_dataset(
            10000,
            1000,
            10,
            self.tempdir_name,
            data_seed = 5,
            num_neighbors = 128,
            query_seed = 1000,
        )

        # Make sure the required components exist.
        data_file = os.path.join(self.tempdir_name, "data.fvecs")
        query_file = os.path.join(self.tempdir_name, "queries.fvecs")
        groundtruth_file = os.path.join(self.tempdir_name, "groundtruth.ivecs")

        self.assertTrue(os.path.isfile(data_file))
        self.assertTrue(os.path.isfile(query_file))
        self.assertTrue(os.path.isfile(groundtruth_file))

        data = pysvs.read_vecs(data_file)
        self.assertEqual(data.ndim, 2)
        self.assertEqual(data.shape[0], 10000)
        self.assertEqual(data.shape[1], 10)
        self.assertEqual(data.dtype, np.float32)

        queries = pysvs.read_vecs(query_file)
        self.assertEqual(queries.ndim, 2)
        self.assertEqual(queries.shape[0], 1000)
        self.assertEqual(queries.shape[1], 10)
        self.assertEqual(queries.dtype, np.float32)

        groundtruth = pysvs.read_vecs(groundtruth_file)
        self.assertEqual(groundtruth.ndim, 2)
        self.assertEqual(groundtruth.shape[0], 1000)
        self.assertEqual(groundtruth.shape[1], 128)
        self.assertEqual(groundtruth.dtype, np.uint32)

        # Manually compute the ground truth for the queries and data.
        # Oh the joys of numpy array manipulation.
        M = queries.shape[0]
        N = data.shape[0]
        query_norms = (queries * queries).sum(axis = 1).reshape((M, 1)) * np.ones(shape = (1, N))
        data_norms = (data * data).sum(axis = 1) * np.ones(shape = (M, 1))
        dist_squared = data_norms + query_norms - 2 * queries.dot(data.T)
        print(f"Shape = {dist_squared.shape}")

        k = groundtruth.shape[1]
        k_smallest = np.argsort(dist_squared)[:, :k]

        # Some ties may not be resolved correctly, so provide some wiggle room.
        # Older architectures with different distance implementations may resolve ties
        # differently.
        #
        # Set max-ties to 6 to handle these older architectures as well (on newer
        # architecture, it's generally 4).
        max_ties = 6
        expected_equal_lower = groundtruth.size - max_ties
        actually_equal = np.count_nonzero(k_smallest == groundtruth)

        if actually_equal < expected_equal_lower:
            print(f"Expected {expected_equal_lower} number of entries. Instead, got {actually_equal}!")

        self.assertLessEqual(expected_equal_lower, actually_equal)

