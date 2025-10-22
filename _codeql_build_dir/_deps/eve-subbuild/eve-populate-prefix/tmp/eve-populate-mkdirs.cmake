# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/runner/work/ScalableVectorSearch/ScalableVectorSearch/_codeql_build_dir/_deps/eve-src")
  file(MAKE_DIRECTORY "/home/runner/work/ScalableVectorSearch/ScalableVectorSearch/_codeql_build_dir/_deps/eve-src")
endif()
file(MAKE_DIRECTORY
  "/home/runner/work/ScalableVectorSearch/ScalableVectorSearch/_codeql_build_dir/_deps/eve-build"
  "/home/runner/work/ScalableVectorSearch/ScalableVectorSearch/_codeql_build_dir/_deps/eve-subbuild/eve-populate-prefix"
  "/home/runner/work/ScalableVectorSearch/ScalableVectorSearch/_codeql_build_dir/_deps/eve-subbuild/eve-populate-prefix/tmp"
  "/home/runner/work/ScalableVectorSearch/ScalableVectorSearch/_codeql_build_dir/_deps/eve-subbuild/eve-populate-prefix/src/eve-populate-stamp"
  "/home/runner/work/ScalableVectorSearch/ScalableVectorSearch/_codeql_build_dir/_deps/eve-subbuild/eve-populate-prefix/src"
  "/home/runner/work/ScalableVectorSearch/ScalableVectorSearch/_codeql_build_dir/_deps/eve-subbuild/eve-populate-prefix/src/eve-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/runner/work/ScalableVectorSearch/ScalableVectorSearch/_codeql_build_dir/_deps/eve-subbuild/eve-populate-prefix/src/eve-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/runner/work/ScalableVectorSearch/ScalableVectorSearch/_codeql_build_dir/_deps/eve-subbuild/eve-populate-prefix/src/eve-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
