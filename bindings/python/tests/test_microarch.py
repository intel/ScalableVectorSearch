# Copyright 2025 Intel Corporation
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

import unittest
import svs
import archspec.cpu as cpu
import os

class MicroarchTester(unittest.TestCase):
    def test_microarch(self):
        supported_microarchs = svs.microarch.supported
        # Will be set in dispatcher pipeline
        archspec_host_name = os.environ.get("SDE_FLAG")
        if not archspec_host_name:
            archspec_host_name = cpu.host().name
        mapping = {
            "nhm": "nehalem",
            "hsw": "haswell",
            "skx": "skylake_avx512",
            "clx": "cascadelake",
            "icl": "icelake_client",
            "icelake": "icelake_client",
            "spr": "sapphirerapids",
        }
        archspec_host_name = mapping.get(archspec_host_name, archspec_host_name)

        if archspec_host_name in supported_microarchs:
            self.assertTrue(archspec_host_name == svs.microarch.current)
