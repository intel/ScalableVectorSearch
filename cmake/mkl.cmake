# Using single threaded MKL to minimize the interference with SVS ThreadPool
set(MKL_THREADING sequential)

# As a side-effect of `find_package`, the variable `MKL_ROOT` will be defined.
# This helps us find the custom shared-object building if needed.
find_package(MKL CONFIG REQUIRED)

# The custom MKL flow uses MKL's builder utility to create a small shared-library with
# just the symbols used by SVS.
#
# The resulting object and linking are portable and should be suitable for distribution.
if (SVS_EXPERIMENTAL_BUILD_CUSTOM_MKL)
    set(SVS_MKL_CUSTOM_LIBRARY_NAME libmkl_custom)
    set(SVS_MKL_CUSTOM_SO_NAME ${SVS_MKL_CUSTOM_LIBRARY_NAME}.so)
    set(SVS_MKL_CUSTOM_FULL_PATH ${CMAKE_CURRENT_BINARY_DIR}/${SVS_MKL_CUSTOM_SO_NAME})

    # This command creates the custom shared-object that will be linked to.
    add_custom_command(
        OUTPUT ${SVS_MKL_CUSTOM_FULL_PATH}
        COMMAND
            "make"
            "-C"
            "${MKL_ROOT}/tools/builder"
            "libintel64"
            "interface=ilp64"
            "threading=sequential"
            "export=${CMAKE_CURRENT_LIST_DIR}/mkl_functions"
            "MKLROOT=${MKL_ROOT}"
            "name=${CMAKE_CURRENT_BINARY_DIR}/${SVS_MKL_CUSTOM_LIBRARY_NAME}"
    )

    # Create a target for the newly created shared object.
    add_custom_target(svs_mkl_target DEPENDS ${SVS_MKL_CUSTOM_FULL_PATH})

    # Create an imported object for the custom MKL library - configure it to depend on
    # the custom library built.
    add_library(svs_mkl SHARED IMPORTED)
    add_dependencies(svs_mkl svs_mkl_target)
    set_target_properties(
        svs_mkl
        PROPERTIES
            IMPORTED_LOCATION ${SVS_MKL_CUSTOM_FULL_PATH}
            IMPORTED_NO_SONAME TRUE
            INTERFACE_COMPILE_OPTIONS $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>
            INTERFACE_INCLUDE_DIRECTORIES
                $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
    )
    target_link_libraries(${SVS_LIB} INTERFACE svs_mkl)

    # Ensure that the custom MKL library is bundled with the rest of the library.
    include(GNUInstallDirs)
    install(IMPORTED_RUNTIME_ARTIFACTS svs_mkl LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
else()
    target_compile_options(
        ${SVS_LIB} INTERFACE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>
    )
    target_include_directories(
        ${SVS_LIB} INTERFACE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
    )
    target_link_libraries(${SVS_LIB} INTERFACE $<LINK_ONLY:MKL::MKL>)
endif()
