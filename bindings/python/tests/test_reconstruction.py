# Copyright 2024 Intel Corporation
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
    test_vamana_config

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
