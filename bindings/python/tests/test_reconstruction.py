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
    isapprox, \
    test_data_svs, \
    test_data_vecs, \
    test_data_dims, \
    test_graph, \
    test_vamana_config, \
    test_close_lvq

DEBUG = False;

class ReconstructionTester(unittest.TestCase):
    """
    Test the reconstruction interface for indexex.
    """
    def _get_loaders(self, loader: svs.VectorDataLoader):
        sequential = svs.LVQStrategy.Sequential
        turbo = svs.LVQStrategy.Turbo

        return [
            # Uncompressed
            loader,
            # LVQ
            svs.LVQLoader(loader, primary = 8, padding = 0),
            svs.LVQLoader(loader, primary = 4, padding = 0),
            svs.LVQLoader(
                loader, primary = 4, residual = 4, strategy = sequential, padding = 0
            ),
            svs.LVQLoader(
                loader, primary = 4, residual = 4, strategy = turbo, padding = 0
            ),
            svs.LVQLoader(
                loader, primary = 4, residual = 8, strategy = sequential, padding = 0
            ),
            svs.LVQLoader(
                loader, primary = 4, residual = 8, strategy = turbo, padding = 0
            ),
            svs.LVQLoader(loader, primary = 8, residual = 8, padding = 0),

            # LeanVec
            svs.LeanVecLoader(
                loader,
                leanvec_dims = 64,
                primary_kind = svs.LeanVecKind.float32,
                secondary_kind = svs.LeanVecKind.float32,
            ),
            svs.LeanVecLoader(
                loader,
                leanvec_dims = 64,
                primary_kind = svs.LeanVecKind.lvq4,
                secondary_kind = svs.LeanVecKind.lvq8,
                alignment = 0
            ),
            svs.LeanVecLoader(
                loader,
                leanvec_dims = 64,
                primary_kind = svs.LeanVecKind.lvq8,
                secondary_kind = svs.LeanVecKind.lvq8,
                alignment = 0
            ),
            svs.LeanVecLoader(
                loader,
                leanvec_dims = 64,
                primary_kind = svs.LeanVecKind.lvq8,
                secondary_kind = svs.LeanVecKind.float16,
                alignment = 0
            ),
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

    def _compare_lvq(self, data, reconstructed, loader: svs.LVQLoader):
        print(f"LVQ: primary = {loader.primary_bits}, residual = {loader.residual_bits}")
        self.assertTrue(isinstance(loader, svs.LVQLoader))
        self.assertTrue(test_close_lvq(
            data,
            reconstructed,
            primary_bits = loader.primary_bits,
            residual_bits = loader.residual_bits
        ))

    def _compare_leanvec(self, data, reconstructed, loader: svs.LeanVecLoader):
        self.assertTrue(isinstance(loader, svs.LeanVecLoader))
        secondary_kind = loader.secondary_kind
        if secondary_kind == svs.LeanVecKind.float32:
            self.assertTrue(np.array_equal(data, reconstructed))
        elif secondary_kind == svs.LeanVecKind.float16:
            self.assertTrue(np.allclose(data, reconstructed))
        elif secondary_kind == svs.LeanVecKind.lvq4:
            self.assertTrue(test_close_lvq(data, reconstructed, primary_bits = 4))
        elif secondary_kind == svs.LeanVecKind.lvq8:
            self.assertTrue(test_close_lvq(data, reconstructed, primary_bits = 8))
        else:
            raise Exception(f"Unknown leanvec kind {secondary_kind}")

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
            elif isinstance(loader, svs.LVQLoader):
                self._compare_lvq(shuffled_data, r, loader)
            elif isinstance(loader, svs.LeanVecLoader):
                self._compare_leanvec(shuffled_data, r, loader)
            else:
                raise Exception(f"Unhandled loader kind: {loader}")

