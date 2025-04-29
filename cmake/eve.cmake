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

Include(FetchContent)
FetchContent_Declare(
    eve
    GIT_REPOSITORY https://github.com/jfalcou/eve
    GIT_TAG v2023.02.15
)

# Set EVE-specific CMake options *before* FetchContent_MakeAvailable
# Force the architecture detection for GCC on Linux ARM (AArch64)
# This should bypass problematic internal detection.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" AND
   (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR NOT CMAKE_CXX_COMPILER_ID) # Check early
  )
    message(STATUS "ARM Linux GCC detected: Forcing EVE architecture via EVE_TARGET_ARCH=arm_v8")
    # Set the CMake variable EVE uses to identify the target architecture.
    # 'arm_v8' typically implies NEON/ASIMD support within EVE's logic.
    # Check EVE documentation/CMake files for the exact variable name if this doesn't work.
    set(EVE_TARGET_ARCH "arm_v8" CACHE STRING "Force EVE target architecture" FORCE)

    # Also ensure the compiler itself targets a compatible architecture
    if(NOT CMAKE_CXX_FLAGS MATCHES "-march=")
       string(APPEND CMAKE_CXX_FLAGS_INIT " -march=armv8-a")
       string(APPEND CMAKE_C_FLAGS_INIT " -march=armv8-a")
       # Update the cache variables as well, just in case
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a" CACHE STRING "CXX Flags" FORCE)
       set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a" CACHE STRING "C Flags" FORCE)
    endif()

endif()

set(EVE_BUILD_TEST OFF)
FetchContent_MakeAvailable(eve)
target_link_libraries(${SVS_LIB} INTERFACE eve::eve)
