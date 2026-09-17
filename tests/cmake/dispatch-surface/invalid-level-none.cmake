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

# NONE means "no level is present"; it has no translation unit and no object
# library, so a row for it names files that must not exist.
# EXPECT-ERROR: ISA level 'NONE' in SVS_ISA_LEVELS is not declarable
set(SVS_SUPPORTED_DIMS 128)
set(SVS_ISA_LEVELS
    "NONE|haswell|avx2"
    "AVX512|skylake-avx512|avx512"
)
