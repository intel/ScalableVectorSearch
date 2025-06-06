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

function(create_utility exe file)
    add_executable(${exe} ${file})
    target_include_directories(
        ${exe}
        PRIVATE ${CMAKE_CURRENT_LIST_DIR}
    )
    # Get header directory and C++ 20
    target_link_libraries(${exe} PRIVATE svs::svs)

    # Get common compiler options with the unit tests.
    target_link_libraries(${exe} PRIVATE svs_compile_options svs_microarch_options_base)

    # Link with third-party executables.
    target_link_libraries(${exe} PRIVATE fmt::fmt)

    # Optionally apply clang-tidy to the executable
    if (SVS_EXPERIMENTAL_CLANG_TIDY AND CLANG_TIDY_COMMAND)
        set_target_properties(
            ${exe}
            PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}"
        )
    endif()
endfunction()

create_utility(graph_stat graph_stat.cpp)

# Legacy conversion routines.
create_utility(convert_legacy convert_legacy.cpp)
create_utility(assemble_vamana assemble_vamana.cpp)
create_utility(upgrade_vamana_index_parameters upgrade_vamana_index_parameters.cpp)

create_utility(build_index build_index.cpp)
create_utility(compute_recall compute_recall.cpp)
create_utility(convert_data_to_float16 convert_data_to_float16.cpp)
create_utility(search_index search_index.cpp)
create_utility(consolidate characterization/consolidate.cpp)
create_utility(logging logging.cpp)
# create_utility(mutable characterization/mutable.cpp)

# # Benchmark
# create_utility(benchmark_index_build benchmarks/index_build.cpp)

svs_register_static_dimensions(96)
