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

from tempfile import TemporaryDirectory

import pysvs

class VamanaCommonTester(unittest.TestCase):
    def test_search_buffer_config(self):
        x = pysvs.SearchBufferConfig()
        self.assertEqual(x.search_window_size, 0)
        self.assertEqual(x.search_buffer_capacity, 0)
        self.assertEqual(x, pysvs.SearchBufferConfig())

        x = pysvs.SearchBufferConfig(10)
        self.assertEqual(x.search_window_size, 10)
        self.assertEqual(x.search_buffer_capacity, 10)
        self.assertNotEqual(x, pysvs.SearchBufferConfig())
        self.assertEqual(x, pysvs.SearchBufferConfig(10))

        x = pysvs.SearchBufferConfig(10, 20)
        self.assertEqual(x.search_window_size, 10)
        self.assertEqual(x.search_buffer_capacity, 20)
        self.assertNotEqual(x, pysvs.SearchBufferConfig())
        self.assertNotEqual(x, pysvs.SearchBufferConfig(10))
        self.assertEqual(x, pysvs.SearchBufferConfig(10, 20))

    def test_vamana_search_parameters(self):
        # Default definitions.
        x = pysvs.VamanaSearchParameters()
        self.assertEqual(x.buffer_config, pysvs.SearchBufferConfig())
        self.assertEqual(x.search_buffer_visited_set, False)
        self.assertEqual(x, pysvs.VamanaSearchParameters())

        x.buffer_config = pysvs.SearchBufferConfig(20)
        self.assertEqual(x.buffer_config, pysvs.SearchBufferConfig(20))
        x.search_buffer_visited_set = True
        self.assertEqual(x.search_buffer_visited_set, True)
        self.assertNotEqual(x, pysvs.VamanaSearchParameters())
        self.assertEqual(x, pysvs.VamanaSearchParameters(pysvs.SearchBufferConfig(20), True))

        x = pysvs.VamanaSearchParameters(pysvs.SearchBufferConfig(10, 20))
        self.assertEqual(x.buffer_config, pysvs.SearchBufferConfig(10, 20))
        self.assertEqual(x.search_buffer_visited_set, False)
        self.assertNotEqual(x, pysvs.VamanaSearchParameters())
        self.assertEqual(x, pysvs.VamanaSearchParameters(pysvs.SearchBufferConfig(10, 20)))

        x = pysvs.VamanaSearchParameters(pysvs.SearchBufferConfig(10, 20), True)
        self.assertEqual(x.buffer_config, pysvs.SearchBufferConfig(10, 20))
        self.assertEqual(x.search_buffer_visited_set, True)

