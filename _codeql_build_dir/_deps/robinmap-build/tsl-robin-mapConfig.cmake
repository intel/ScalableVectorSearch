# This module sets the following variables:
# * tsl-robin-map_FOUND - true if tsl-robin-map found on the system
# * tsl-robin-map_INCLUDE_DIRS - the directory containing tsl-robin-map headers

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was tsl-robin-mapConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

if(NOT TARGET tsl::robin_map)
  include("${CMAKE_CURRENT_LIST_DIR}/tsl-robin-mapTargets.cmake")
  get_target_property(tsl-robin-map_INCLUDE_DIRS tsl::robin_map INTERFACE_INCLUDE_DIRECTORIES)
endif()
