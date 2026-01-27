#!/usr/bin/env bash
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


# Allow users to supply a custom path to `clang-format`
CLANGFORMAT="${1:-clang-format}"
DIRECTORIES=( "bindings/python/src" "bindings/python/include" "bindings/cpp" "include" "benchmark" "tests" "utils" "examples/cpp" )

for i in "${DIRECTORIES[@]}"
do
    find "./$i" -maxdepth 1 \( -iname "*.h" -o -iname "*.cpp" \) ! -iname "*toml_impl.h" | xargs "$CLANGFORMAT" -i
done
