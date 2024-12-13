# Copyright 2023 Intel Corporation
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

# Dynamic loading logic.
from .loader import library, current_backend, available_backends

# Reexport all public functions and structs from the inner module.
lib = library()
globals().update(
    {k : v for (k, v) in lib.__dict__.items() if not k.startswith("__")}
)

# Misc types and functions
from .common import \
    np_to_svs, \
    read_npy, \
    read_vecs, \
    write_vecs, \
    read_svs, \
    k_recall_at, \
    generate_test_dataset

# Make the upgrader available without explicit import.
from . import upgrader

