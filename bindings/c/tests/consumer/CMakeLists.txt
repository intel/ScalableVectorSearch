# Copyright 2026 Intel Corporation
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

# Standalone project that consumes the *installed* C API package the same way a
# downstream integration would. It is deliberately not part of the C API build:
# it must be configured against an install tree so that a broken exported
# target (a missing find_dependency, or a C++ requirement leaking onto a C
# consumer) fails here instead of in the downstream project.
cmake_minimum_required(VERSION 3.21)
project(svs_c_api_consumer LANGUAGES C)

find_package(svs_c_api REQUIRED)

add_executable(c_api_consumer main.c)
target_link_libraries(c_api_consumer PRIVATE svs::svs_c_api)
