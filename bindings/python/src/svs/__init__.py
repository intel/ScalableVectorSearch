#
# Copyright (C) 2023, Intel Corporation
#
# You can redistribute and/or modify this software under the terms of the
# GNU Affero General Public License version 3.
#
# You should have received a copy of the GNU Affero General Public License
# version 3 along with this software. If not, see
# <https://www.gnu.org/licenses/agpl-3.0.en.html>.
#

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

