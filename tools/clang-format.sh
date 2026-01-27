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
STAGED_ONLY="${2:-false}"
DIRECTORIES=( "bindings/python/src" "bindings/python/include" "bindings/cpp" "include" "benchmark" "tests" "utils" "examples/cpp" )

for i in "${DIRECTORIES[@]}"
do
    if [[ "$STAGED_ONLY" == "true" ]]; then
        git diff --cached --name-only --diff-filter=ACM | grep -E "^$i/.*\.(h|cpp)$" | grep -v "toml_impl.h" | xargs -r "$CLANGFORMAT" -i
    else
        find "./$i" \( -iname "*.h" -o -iname "*.cpp" \) ! -iname "*toml_impl.h" -print0 | xargs -n1 -0 "$CLANGFORMAT" -i
    fi
done
