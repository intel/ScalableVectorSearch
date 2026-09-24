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

#####
##### The dispatch surface for the x86 distance kernels.
#####
##### This file is the single place the extent list and the ISA levels are
##### written down. Everything derived from them is generated:
#####
#####   - svs/core/distance/dispatch_surface.h, which drives every `extern
#####     template` and explicit instantiation, and `supported_dim_list`
#####   - the object library each ISA level's translation unit is compiled into,
#####     and the instruction budget it is compiled at
#####
##### Edit this file. The header is not in the source tree at all: it is written
##### into the build directory on every configure and installed from there.
#####
##### A build may point somewhere else with -DSVS_DISPATCH_SURFACE_FILE=<file>,
##### in which case only that build tree describes the overridden surface.
#####
##### Bookkeeping
#####
##### To add or remove a fixed extent:
#####   Edit `SVS_SUPPORTED_DIMS`. The generated header, every `extern
#####   template`, and `supported_dim_list` follow automatically.
#####
##### To add an ISA level:
#####   1. Add a row to `SVS_ISA_LEVELS`.
#####   2. Add a `SVS_TYPE_PAIRS_<enumerator>` list in
#####      include/svs/multi-arch/x86/preprocessor.h, listing the element-type
#####      pairs that level has kernels for.
#####   3. Add the level's translation unit at
#####      include/svs/multi-arch/x86/<infix>.cpp.
#####   The object library and its compile flags follow from the row here. A
#####   level without a type-pair list is a compile error, not an empty
#####   instantiation set.
#####
##### To add or remove an element-type pair for a level:
#####   Edit that level's `SVS_TYPE_PAIRS_<enumerator>` list in
#####   include/svs/multi-arch/x86/preprocessor.h. Not configured here: a type
#####   pair exists because an implementation exists for it.
#####
##### To change a level's instruction budget:
#####   Edit the middle field of its row in `SVS_ISA_LEVELS`, observing the
#####   constraint recorded there.
#####


##
## List of supported dimensions; kernels will be available for these dimensions
## If used with dimensions not in this list, compute will be dispatched to
## svs::Dynamic, which is automatically added to the list of supported dims.
##

set(SVS_SUPPORTED_DIMS 64 96 100 128 160 200 512 768)

##
## ISA mapping
##
## All compute kernels will be compiled for the architectures below
## NONE is not added; it only serves as a fallback and it will compile with
## the flags set by the consumer.
## Format is "ENUM|arch|infix", where
##   - ENUM is the AVX_AVAILABILITY value from distance_core.h
##   - arch is the -march flag passed to the compiler
##   - infix is a **unique** string used for object and library names

set(SVS_ISA_LEVELS
    "AVX2|haswell|avx2"
    "AVX512|cascadelake|avx512"
)
