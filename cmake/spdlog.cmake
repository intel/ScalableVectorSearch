# Copyright 2024 Intel Corporation
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

# Fetch `spdlog`
# Configure it to use our version of `fmtlib`.
# Since we consume spdlog

# TODO: We need a better way of doing dependency management and potentially linking
# against system libraries instead of always downloading.
#
# As far as I can tell, libraries compiled with C++11 *should* be ABI compatible with
# those compiled with C++20 (modulo compiler bugs).
#
# However, if we can head off some of those issues, might as well.
set(PRESET_CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD ${SVS_CXX_STANDARD})

set(SPDLOG_INSTALL YES CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL YES CACHE BOOL "" FORCE)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG v1.15.3
)
FetchContent_MakeAvailable(spdlog)
target_link_libraries(${SVS_LIB} INTERFACE spdlog::spdlog)

set(CMAKE_CXX_STANDARD ${PRESET_CMAKE_CXX_STANDARD})
