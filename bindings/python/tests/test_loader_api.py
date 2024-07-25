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
import itertools
import svs

# Local dependencies
from .common import test_data_vecs

DEBUG = False;

class LoaderAPITester(unittest.TestCase):
    """
    Test routines for the various loader classes.
    """
    def _get_basic_loader(self):
        loader = svs.VectorDataLoader(test_data_vecs, data_type = svs.float32)
        self.assertEqual(loader.data_type, svs.float32)
        self.assertEqual(loader.dims, 128)
        return loader
