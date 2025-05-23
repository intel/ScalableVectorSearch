#!/bin/bash
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

# Script to get CPU information using platform-agnostic python packages

# Install python packages if not present in the environment
if ! python -m pip show archspec > /dev/null 2>&1; then
    python -m pip install archspec
fi

if ! python -m pip show py-cpuinfo > /dev/null 2>&1; then
    python -m pip install py-cpuinfo
fi

# Print host microarchitecture
python -c "import archspec.cpu; \
    print('Host Microarchitecture[archspec]:', archspec.cpu.host().name)"

# Print full CPU information
python -c "import pprint, cpuinfo; \
    print('CPU info[py-cpuinfo]:'); \
    pprint.pprint(cpuinfo.get_cpu_info(), indent=4, compact=True)"
