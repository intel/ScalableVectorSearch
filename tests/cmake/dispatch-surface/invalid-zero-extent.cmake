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

# A zero-length vector has no kernel to compile.
# EXPECT-ERROR: SVS_SUPPORTED_DIMS contains '0'
set(SVS_SUPPORTED_DIMS 0 128)
set(SVS_ISA_LEVELS "AVX2|haswell|avx2")
