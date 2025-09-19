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

cmake_minimum_required(VERSION 3.21)

project(svs_shared_library_example
    LANGUAGES CXX
)

# Other AVX versions can be found at https://github.com/intel/ScalableVectorSearch/releases.
set(SVS_URL "https://github.com/intel/ScalableVectorSearch/releases/download/v0.0.9/svs-shared-library-0.0.9.tar.gz")

include(FetchContent)
FetchContent_Declare(
  svs
  URL "${SVS_URL}"
)
FetchContent_MakeAvailable(svs)

list(APPEND CMAKE_PREFIX_PATH "${svs_SOURCE_DIR}")
find_package(svs REQUIRED)
find_library(SVS_SHARED svs_shared_library)

set(SVS_CXX_STANDARD 20)
SET(CMAKE_CXX_FLAGS  "-O3 -DNDEBUG -std=gnu++20 -march=native -mtune=native -Werror -Wall -Wextra -Wpedantic" )

# Configure path to the datasets used in these examples
set(DATA_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../../data/test_dataset")

function(create_example_executable exe file)
    add_executable(${exe} ${file})
    target_compile_definitions(${exe} PRIVATE SVS_DATA_DIR="${DATA_DIRECTORY}")
    target_link_libraries(${exe} PUBLIC ${SVS_SHARED} svs::svs)
endfunction()

create_example_executable(shared shared.cpp)
create_example_executable(example_vamana_with_compression_lvq example_vamana_with_compression_lvq.cpp)
create_example_executable(example_vamana_with_compression example_vamana_with_compression.cpp)
create_example_executable(example_vamana_with_compression_dynamic example_vamana_with_compression_dynamic.cpp)
