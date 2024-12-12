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
