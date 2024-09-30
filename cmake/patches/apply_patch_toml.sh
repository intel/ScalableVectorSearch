# Copyright (C) 2024 Intel Corporation
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

TOML_PATCH=$1
# First check if the patch can be reversed (i.e. was it applied earlier)
# Apply it only when it's not applied earlier
if ! git apply -R --ignore-whitespace ${TOML_PATCH} --check; then git apply --ignore-whitespace ${TOML_PATCH}; fi
