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
##### Derives the dispatch surface declared in cmake/dispatch-surface.cmake.
#####
##### Defines svs_generate_dispatch_surface(), which produces:
#####   <build>/generated/include/svs/core/distance/dispatch_surface.h  (build tree only)
#####   SVS_DISPATCH_TU_SPECS -- "<src>|<level>|<arch>|<infix>", one per ISA level
#####
##### cmake/multi-arch.cmake calls the function and receives both.
#####

include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/dispatch-levels.cmake")

set(SVS_DISPATCH_GEN_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")

set(SVS_DEFAULT_DISPATCH_SURFACE_FILE "${CMAKE_CURRENT_LIST_DIR}/dispatch-surface.cmake")
set(SVS_DISPATCH_SURFACE_FILE "${SVS_DEFAULT_DISPATCH_SURFACE_FILE}"
    CACHE FILEPATH
    "Declaration of the ahead-of-time distance-kernel dispatch surface"
)

#####
##### Read the declaration
#####

function(svs_dispatch_read_declaration surface_file x86_src_dir
         out_supported_dims out_dim_list out_dim_count out_isa_levels)
    set(SVS_DISPATCH_SURFACE_FILE "${surface_file}")
    set(SVS_X86_SRC_DIR "${x86_src_dir}")
    set(${out_supported_dims} "${SVS_SUPPORTED_DIMS}" PARENT_SCOPE)
    set(${out_dim_list} "${SVS_DIM_LIST}" PARENT_SCOPE)
    set(${out_dim_count} "${SVS_DIM_COUNT}" PARENT_SCOPE)
    set(${out_isa_levels} "${SVS_ISA_LEVELS}" PARENT_SCOPE)
endfunction()

#####
##### Build the macro bodies
#####

function(svs_dispatch_build_expansions dim_list isa_levels
         out_dim_loop out_target_loop out_level_loop out_level_defines)
    set(dim_loop "\\\n")
    foreach(dim IN LISTS dim_list)
        string(APPEND dim_loop "    M(${dim}) \\\n")
    endforeach()
    string(APPEND dim_loop "    /* end */")

    set(target_loop "\\\n")
    set(level_loop "\\\n")
    set(level_defines "")
    foreach(level_spec IN LISTS isa_levels)
        svs_parse_isa_level("${level_spec}" level _ _)
        string(APPEND level_loop "    M(${level}) \\\n")
        string(APPEND level_defines "#define SVS_ISA_LEVEL_${level} 1\n")
        foreach(dim IN LISTS dim_list)
            string(APPEND target_loop "    M(${dim}, ${level}) \\\n")
        endforeach()
    endforeach()
    string(APPEND target_loop "    /* end */")
    string(APPEND level_loop "    /* end */")
    string(STRIP "${level_defines}" level_defines)

    set(${out_dim_loop} "${dim_loop}" PARENT_SCOPE)
    set(${out_target_loop} "${target_loop}" PARENT_SCOPE)
    set(${out_level_loop} "${level_loop}" PARENT_SCOPE)
    set(${out_level_defines} "${level_defines}" PARENT_SCOPE)
endfunction()

#####
##### Derive the translation-unit specs
#####

function(svs_dispatch_derive_tu_specs isa_levels x86_src_dir
         out_tu_specs out_levels out_level_report)
    set(tu_specs)
    set(levels)
    set(level_report)
    foreach(level_spec IN LISTS isa_levels)
        svs_parse_isa_level("${level_spec}" level arch infix)
        list(APPEND tu_specs "${x86_src_dir}/${infix}.cpp|${level}|${arch}|${infix}")
        list(APPEND levels "${level}")
        list(APPEND level_report "AVX_AVAILABILITY::${level} -march=${arch} ${infix}.cpp")
    endforeach()
    set(${out_tu_specs} "${tu_specs}" PARENT_SCOPE)
    set(${out_levels} "${levels}" PARENT_SCOPE)
    set(${out_level_report} "${level_report}" PARENT_SCOPE)
endfunction()

#####
##### Emit the header
#####

# Writes the build-tree header from the template and the expansion strings above.
function(svs_dispatch_emit_header template_file dim_count dim_loop target_loop
         level_loop level_defines include_dir out_header)
    set(SVS_GEN_DIM_COUNT "${dim_count}")
    set(SVS_GEN_DIM_LOOP "${dim_loop}")
    set(SVS_GEN_TARGET_LOOP "${target_loop}")
    set(SVS_GEN_LEVEL_LOOP "${level_loop}")
    set(SVS_GEN_LEVEL_DEFINES "${level_defines}")
    set(header "${include_dir}/svs/core/distance/dispatch_surface.h")
    configure_file("${template_file}" "${header}" @ONLY)
    set(${out_header} "${header}" PARENT_SCOPE)
endfunction()

#####
##### Emit the manifest
#####

function(svs_dispatch_emit_manifest supported_dims levels manifest_file)
    string(REPLACE ";" " " levels_text "${levels}")
    string(REPLACE ";" " " extents_text "${supported_dims}")
    file(GENERATE
        OUTPUT "${manifest_file}"
        CONTENT "set(SVS_MANIFEST_FIXED_EXTENTS ${extents_text})
set(SVS_MANIFEST_LEVELS ${levels_text})
"
    )
endfunction()

#####
##### Orchestrate
#####

# Reads the declaration, writes the build-tree header and the ctest manifest, and
# reports the surface. Returns the TU specs and the generated header path, which is
# how cmake/multi-arch.cmake receives them; nothing here escapes via a bare
# file-scope set().
function(svs_generate_dispatch_surface out_tu_specs out_header)
    set(x86_src_dir "${PROJECT_SOURCE_DIR}/include/svs/multi-arch/x86")

    svs_dispatch_read_declaration(
        "${SVS_DISPATCH_SURFACE_FILE}" "${x86_src_dir}"
        supported_dims dim_list dim_count isa_levels
    )

    # Re-run configure when the declaration changes, so the generated header and
    # the translation units cannot go stale.
    set_property(
        DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${SVS_DISPATCH_SURFACE_FILE}"
    )

    svs_dispatch_build_expansions(
        "${dim_list}" "${isa_levels}"
        dim_loop target_loop level_loop level_defines
    )
    svs_dispatch_derive_tu_specs(
        "${isa_levels}" "${x86_src_dir}"
        tu_specs levels level_report
    )

    # The build always compiles against the build-tree copy, and it is placed
    # ahead of the source include directory so that it wins.
    set(generated_include_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
    svs_dispatch_emit_header(
        "${SVS_DISPATCH_GEN_CMAKE_DIR}/templates/dispatch_surface.h.in"
        "${dim_count}" "${dim_loop}" "${target_loop}" "${level_loop}" "${level_defines}"
        "${generated_include_dir}"
        header
    )
    target_include_directories(
        ${SVS_LIB} BEFORE INTERFACE $<BUILD_INTERFACE:${generated_include_dir}>
    )

    svs_dispatch_emit_manifest(
        "${supported_dims}" "${levels}"
        "${CMAKE_BINARY_DIR}/dispatch_surface.manifest.cmake"
    )

    #####
    ##### Report the surface
    #####

    list(LENGTH isa_levels level_count)
    string(REPLACE ";" " " dims_display "${supported_dims}")
    message(STATUS "Dispatch surface: ${dim_count} extents x ${level_count} ISA levels")
    message(STATUS "  extents: ${dims_display} svs::Dynamic")
    foreach(entry IN LISTS level_report)
        message(STATUS "  level:   ${entry}")
    endforeach()

    # Every enumerator without a translation unit is still reachable -- the entry
    # points fall back to it -- so its kernels are built by each consumer instead.
    set(enum_header "${PROJECT_SOURCE_DIR}/include/svs/core/distance/distance_core.h")
    if(EXISTS "${enum_header}")
        file(READ "${enum_header}" enum_text)
        if(enum_text MATCHES "enum class AVX_AVAILABILITY[ \t\r\n]*{([^}]*)}")
            string(REPLACE "," ";" enumerators "${CMAKE_MATCH_1}")
            set(undeclared)
            foreach(enumerator IN LISTS enumerators)
                string(STRIP "${enumerator}" enumerator)
                if(enumerator AND NOT enumerator IN_LIST levels)
                    list(APPEND undeclared "${enumerator}")
                endif()
            endforeach()
            if(undeclared)
                string(REPLACE ";" ", " undeclared_display "${undeclared}")
                message(STATUS "  not in the surface: ${undeclared_display}")
                message(STATUS
                    "           dispatched to, but compiled by no translation unit, so "
                    "every consumer instantiates those kernels itself, at its own -march"
                )
            endif()
        endif()
    endif()

    set(${out_tu_specs} "${tu_specs}" PARENT_SCOPE)
    set(${out_header} "${header}" PARENT_SCOPE)
endfunction()
