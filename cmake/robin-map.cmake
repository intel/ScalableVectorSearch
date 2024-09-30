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
    RobinMap
    GIT_REPOSITORY https://github.com/Tessil/robin-map
    GIT_TAG v1.0.1
)

FetchContent_MakeAvailable(RobinMap)
target_link_libraries(${SVS_LIB} INTERFACE tsl::robin_map)
