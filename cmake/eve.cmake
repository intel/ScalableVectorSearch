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

set(EVE_BUILD_TEST OFF)
FetchContent_MakeAvailable(eve)
target_link_libraries(${SVS_LIB} INTERFACE eve::eve)
