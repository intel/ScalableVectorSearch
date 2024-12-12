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

from tempfile import TemporaryDirectory

import svs

class VamanaCommonTester(unittest.TestCase):
    def test_search_buffer_config(self):
        x = svs.SearchBufferConfig()
        self.assertEqual(x.search_window_size, 0)
        self.assertEqual(x.search_buffer_capacity, 0)
        self.assertEqual(x, svs.SearchBufferConfig())

        x = svs.SearchBufferConfig(10)
        self.assertEqual(x.search_window_size, 10)
        self.assertEqual(x.search_buffer_capacity, 10)
        self.assertNotEqual(x, svs.SearchBufferConfig())
        self.assertEqual(x, svs.SearchBufferConfig(10))

        x = svs.SearchBufferConfig(10, 20)
        self.assertEqual(x.search_window_size, 10)
        self.assertEqual(x.search_buffer_capacity, 20)
        self.assertNotEqual(x, svs.SearchBufferConfig())
        self.assertNotEqual(x, svs.SearchBufferConfig(10))
        self.assertEqual(x, svs.SearchBufferConfig(10, 20))

    def test_vamana_search_parameters(self):
        # Default definitions.
        x = svs.VamanaSearchParameters()
        self.assertEqual(x.buffer_config, svs.SearchBufferConfig())
        self.assertEqual(x.search_buffer_visited_set, False)
        self.assertEqual(x, svs.VamanaSearchParameters())

        x.buffer_config = svs.SearchBufferConfig(20)
        self.assertEqual(x.buffer_config, svs.SearchBufferConfig(20))
        x.search_buffer_visited_set = True
        self.assertEqual(x.search_buffer_visited_set, True)
        self.assertNotEqual(x, svs.VamanaSearchParameters())
        self.assertEqual(x, svs.VamanaSearchParameters(svs.SearchBufferConfig(20), True))

        x = svs.VamanaSearchParameters(svs.SearchBufferConfig(10, 20))
        self.assertEqual(x.buffer_config, svs.SearchBufferConfig(10, 20))
        self.assertEqual(x.search_buffer_visited_set, False)
        self.assertNotEqual(x, svs.VamanaSearchParameters())
        self.assertEqual(x, svs.VamanaSearchParameters(svs.SearchBufferConfig(10, 20)))

        x = svs.VamanaSearchParameters(svs.SearchBufferConfig(10, 20), True)
        self.assertEqual(x.buffer_config, svs.SearchBufferConfig(10, 20))
        self.assertEqual(x.search_buffer_visited_set, True)

