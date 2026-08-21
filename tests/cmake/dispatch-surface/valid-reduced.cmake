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

# A surface that shares no extent with the default declaration, so a build using
# it cannot accidentally pass by reusing a committed header. Both ISA levels are
# kept so that runtime dispatch is still exercised on an AVX-512 host.
#
# Built and tested by the `non-default surface` CI job.

set(SVS_SUPPORTED_DIMS 32 384)
set(SVS_ISA_LEVELS
    "AVX2|haswell|avx2"
    "AVX512|cascadelake|avx512"
)
