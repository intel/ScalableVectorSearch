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

foreach(SAMPLE_NAME simple save_load)
    set(SAMPLE_TARGET c_api_${SAMPLE_NAME})
    add_executable(${SAMPLE_TARGET} ${SAMPLE_NAME}.c)

    target_link_libraries(${SAMPLE_TARGET} PRIVATE svs_c_api)

    target_include_directories(${SAMPLE_TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
endforeach(SAMPLE_NAME)
