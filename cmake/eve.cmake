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

Include(FetchContent)
FetchContent_Declare(
    eve
    GIT_REPOSITORY https://github.com/jfalcou/eve
    GIT_TAG v2022.09.1
)

set(EVE_BUILD_TEST OFF)
FetchContent_MakeAvailable(eve)
target_link_libraries(${SVS_LIB} INTERFACE eve::eve)
