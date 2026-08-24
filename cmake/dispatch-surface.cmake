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
#####   - include/svs/core/distance/dispatch_surface.h, which drives every
#####     `extern template` and explicit instantiation, and `supported_dim_list`
#####   - the object library each ISA level's translation unit is compiled into,
#####     and the instruction budget it is compiled at
#####
##### Edit this file. Do not edit the generated header; it is regenerated on
##### every configure and your changes there will be overwritten.
#####
##### A build may point somewhere else with -DSVS_DISPATCH_SURFACE_FILE=<file>,
##### in which case the committed header is left alone and only the build tree
##### describes that surface.
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

# Extents that get their own fixed-extent kernel.
#
# This list is a *performance* choice, not a compatibility one. Any
# dimensionality not listed here still works and is fully supported: it
# dispatches to the `svs::Dynamic` kernel, which takes the length at run time.
# Listing an extent buys a fully unrolled kernel for it, at the cost of one more
# set of instantiations across every ISA level and type pair.
#
# `svs::Dynamic` is required and is appended automatically -- do not list it.
set(SVS_SUPPORTED_DIMS 64 96 100 128 160 200 512 768)

# Runtime ISA levels. Each has one translation unit, which instantiates every
# extent above at that level.
#
# Only levels with their own translation unit appear here, which is why
# `AVX_AVAILABILITY::NONE` has no row: nothing instantiates its kernels ahead
# of time, so each translation unit that uses them instantiates them itself, at
# its own `-march` -- generic `x86-64` for this project's own build. A row here
# would name a translation unit and an object library that do not exist.
#
# `AVX_AVAILABILITY` mangles positionally, so `NONE` must keep its enumerator
# value despite having no row; renumbering the enumerators is an ABI break.
#
#   <enumerator>|<instruction budget>|<TU infix>
#
#   enumerator         a value of `svs::distance::AVX_AVAILABILITY`
#   instruction budget the -march this level's kernels are compiled to; it must
#                      match what the level guarantees about the host, because
#                      anything the compiler is allowed to emit here will run on
#                      any host that satisfies the level
#   TU infix           names both the level's translation unit,
#                      include/svs/multi-arch/x86/<infix>.cpp, and the object
#                      library it is compiled into
#
# Adding a level here also requires a `SVS_TYPE_PAIRS_<enumerator>` list in
# include/svs/multi-arch/x86/preprocessor.h, saying which element-type pairs
# that level has kernels for. That is deliberately not configured here: a type
# pair exists because an implementation exists for it.
set(SVS_ISA_LEVELS
    "AVX2|haswell|avx2"
    "AVX512|cascadelake|avx512"
)
