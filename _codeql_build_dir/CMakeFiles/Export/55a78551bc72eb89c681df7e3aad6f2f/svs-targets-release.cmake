#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "svs::svs_x86_objects" for configuration "Release"
set_property(TARGET svs::svs_x86_objects APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(svs::svs_x86_objects PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsvs_x86_objects.a"
  )

list(APPEND _cmake_import_check_targets svs::svs_x86_objects )
list(APPEND _cmake_import_check_files_for_svs::svs_x86_objects "${_IMPORT_PREFIX}/lib/libsvs_x86_objects.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
