#!/bin/sh
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


# Allow users to supply a custom path to `clang-format`
CLANGFORMAT="${1:-clang-format}"
DIRECTORIES=( "bindings/python/src" "bindings/python/include" "include" "benchmark" "tests" "utils" "examples/cpp" )

for i in "${DIRECTORIES[@]}"
do
    find "./$i" \( -iname "*.h" -o -iname "*.cpp" \) ! -iname "*toml_impl.h" | xargs "$CLANGFORMAT" -i
done

