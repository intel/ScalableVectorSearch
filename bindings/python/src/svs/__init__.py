# Copyright (C) 2023 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit
# this software or the related documents without Intel's prior written
# permission.
#
# This software and the related documents are provided as is, with no
# express or implied warranties, other than those that are expressly stated
# in the License.

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

# LeanVec computation
from .leanvec import compute_leanvec_matrices

# Make the upgrader available without explicit import.
from . import upgrader

