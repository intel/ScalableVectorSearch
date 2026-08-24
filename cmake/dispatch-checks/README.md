<!--
  ~ Copyright 2026 Intel Corporation
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# Dispatch surface checks

These scripts are run by ctest to hold the built binary to the surface
declared in `cmake/dispatch-surface.cmake`. They are not part of building
the library. See `cmake/dispatch-surface.cmake` for documentation on the
declaration format, the current extent list and ISA levels, and how to add
a new level.

These tests are only added to the build when the x86 object libraries
exist. Run `ctest` from `<build>/tests`, not from the build root.

## check-dispatch-linkage.cmake

**Test:** `dispatch_surface_linkage`

Verifies that every kernel declared in the surface is defined in the
archive, that the archive defines no undeclared kernels, and that a test
probe naming every kernel defines none of its own. Failure means a
declared kernel is missing or an extra kernel was built but not declared.

## check-dispatch-declaration.cmake

**Test:** `dispatch_surface_declaration`

Verifies that the symbol count derived from the surface declaration matches
the symbol count in the archive and in a consumer object that names every
entry point. Failure means the declaration's extent list, ISA levels, type
pairs, or distance enumerator order disagrees with the built kernels. This
catches generator bugs that would pass linkage checks by being wrong
consistently in both the archive and the probe.

## check-dispatch-instructions.cmake

**Test:** `dispatch_instructions_<infix>` (one test per ISA level)

Verifies that each level's object file stays within its instruction budget,
emitting only instructions the level's runtime predicate guarantees the host
supports. Failure means an object file contains instructions not guaranteed
by its level's predicate, causing illegal-instruction faults on hosts the
dispatcher routes to that level.

## check-dispatch-execution.cmake

**Test:** `dispatch_surface_execution`

Verifies that a call through the entry points on the test host enters the
ISA level the host satisfies, observed by breaking in gdb on each level's
kernel and reporting which one runs. Failure means runtime dispatch routes
to the wrong level, or a specialization disappeared behind a preprocessor
guard while still linking and counting in the symbol table.
