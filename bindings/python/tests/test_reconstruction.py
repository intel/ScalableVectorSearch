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
    test_close_lvq

DEBUG = False;

class ReconstructionTester(unittest.TestCase):
    """
    Test the reconstruction interface for indexex.
    """
    def _get_loaders(self, loader: pysvs.VectorDataLoader):
        sequential = pysvs.LVQStrategy.Sequential
        turbo = pysvs.LVQStrategy.Turbo

        return [
            # Uncompressed
            loader,
            # LVQ
            pysvs.LVQLoader(loader, primary = 8, padding = 0),
            pysvs.LVQLoader(loader, primary = 4, padding = 0),
            pysvs.LVQLoader(
                loader, primary = 4, residual = 8, strategy = sequential, padding = 0
            ),
            pysvs.LVQLoader(
                loader, primary = 4, residual = 8, strategy = turbo, padding = 0
            ),
            pysvs.LVQLoader(loader, primary = 8, residual = 8, padding = 0),

            # LeanVec
            pysvs.LeanVecLoader(
                loader,
                leanvec_dims = 64,
                primary_kind = pysvs.LeanVecKind.float32,
                secondary_kind = pysvs.LeanVecKind.float32,
            ),
            pysvs.LeanVecLoader(
                loader,
                leanvec_dims = 64,
                primary_kind = pysvs.LeanVecKind.lvq8,
                secondary_kind = pysvs.LeanVecKind.lvq8,
                alignment = 0
            ),
            pysvs.LeanVecLoader(
                loader,
                leanvec_dims = 64,
                primary_kind = pysvs.LeanVecKind.lvq8,
                secondary_kind = pysvs.LeanVecKind.float16,
                alignment = 0
            ),
        ]

    def _test_misc(self, loader: pysvs.VectorDataLoader, data):
        num_points = data.shape[0]
        vamana = pysvs.Vamana(test_vamana_config, test_graph, loader)

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

    def _compare_lvq(self, data, reconstructed, loader: pysvs.LVQLoader):
        self.assertTrue(isinstance(loader, pysvs.LVQLoader))
        self.assertTrue(test_close_lvq(
            data,
            reconstructed,
            primary_bits = loader.primary_bits,
            residual_bits = loader.residual_bits
        ))

    def _compare_leanvec(self, data, reconstructed, loader: pysvs.LeanVecLoader):
        self.assertTrue(isinstance(loader, pysvs.LeanVecLoader))
        secondary_kind = loader.secondary_kind
        if secondary_kind == pysvs.LeanVecKind.float32:
            self.assertTrue(np.array_equal(data, reconstructed))
        elif secondary_kind == pysvs.LeanVecKind.float16:
            self.assertTrue(np.allclose(data, reconstructed))
        elif secondary_kind == pysvs.LeanVecKind.lvq4:
            self.assertTrue(test_close_lvq(data, reconstructed, primary_bits = 4))
        elif secondary_kind == pysvs.LeanVecKind.lvq8:
            self.assertTrue(test_close_lvq(data, reconstructed, primary_bits = 8))
        else:
            raise Exception(f"Unknown leanvec kind {secondary_kind}")

    def test_reconstruction(self):
        default_loader = pysvs.VectorDataLoader(test_data_svs, pysvs.DataType.float32)
        all_loaders = self._get_loaders(default_loader)

        data = pysvs.read_vecs(test_data_vecs)

        # Test the error handling separately.
        self._test_misc(default_loader, data)

        all_ids = np.arange(data.shape[0], dtype = np.uint64)
        np.random.shuffle(all_ids)

        shuffled_data = data[all_ids]

        for loader in all_loaders:
            vamana = pysvs.Vamana(test_vamana_config, test_graph, loader)
            r = vamana.reconstruct(all_ids)

            if isinstance(loader, pysvs.VectorDataLoader):
                self.assertTrue(np.array_equal(shuffled_data, r))
            elif isinstance(loader, pysvs.LVQLoader):
                self._compare_lvq(shuffled_data, r, loader)
            elif isinstance(loader, pysvs.LeanVecLoader):
                self._compare_leanvec(shuffled_data, r, loader)
            else:
                raise Exception(f"Unhandled loader kind: {loader}")

