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

# Tests for the Vamana index portion of the SVS module.
import unittest
import os
import warnings
import toml

import numpy as np

from tempfile import TemporaryDirectory

import svs

# Local dependencies
from .common import \
    test_data_svs, \
    test_data_vecs, \
    test_graph, \
    test_vamana_config,

DEBUG = False;

class ReconstructionTester(unittest.TestCase):
    """
    Test the reconstruction interface for indexex.
    """
    def _get_loaders(self, loader: svs.VectorDataLoader):
        return [
            # Uncompressed
            loader,
        ]

    def _test_misc(self, loader: svs.VectorDataLoader, data):
        num_points = data.shape[0]
        vamana = svs.Vamana(test_vamana_config, test_graph, loader)

        # Throw exception on out-of-bounds
        with self.assertRaises(Exception) as context:
            vamana.reconstruct(np.array(num_points))

        # Check that shapes are preserved.
        # 0-D
        d = vamana.dimensions
        self.assertTrue(
            vamana.reconstruct(np.array(0, dtype = np.uint64)).shape == (d,)
        )

        # 1-D
        self.assertTrue(
            vamana.reconstruct(np.zeros(10, dtype = np.uint64)).shape == (10, d)
        )

        # 2-D
        self.assertTrue(
            vamana.reconstruct(np.zeros((10, 10), dtype = np.uint64)).shape == (10, 10, d)
        )

    def test_reconstruction(self):
        default_loader = svs.VectorDataLoader(test_data_svs, svs.DataType.float32)
        all_loaders = self._get_loaders(default_loader)

        data = svs.read_vecs(test_data_vecs)

        # Test the error handling separately.
        self._test_misc(default_loader, data)

        all_ids = np.arange(data.shape[0], dtype = np.uint64)
        np.random.shuffle(all_ids)

        shuffled_data = data[all_ids]

        for loader in all_loaders:
            vamana = svs.Vamana(test_vamana_config, test_graph, loader)
            r = vamana.reconstruct(all_ids)

            if isinstance(loader, svs.VectorDataLoader):
                self.assertTrue(np.array_equal(shuffled_data, r))
            else:
                raise Exception(f"Unhandled loader kind: {loader}")

