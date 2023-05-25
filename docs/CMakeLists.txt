# For some guidance on what's going on here, refer to this blog post:
# https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/

find_package(Doxygen REQUIRED)

# Directory containing C++ examples.
set(EXAMPLE_CPP_DIR "${PROJECT_SOURCE_DIR}/examples/cpp")

# ALL Doxygen specific modifications go here.
# See: https://cmake.org/cmake/help/v3.11/module/FindDoxygen.html

# Generate XML output for consumption by `breathe`.
set(DOXYGEN_GENERATE_XML YES)

# Interpret the "unused" macro as straight-forward pass through.
set(DOXYGEN_ENABLE_PREPROCESSING YES)
set(DOXYGEN_MACRO_EXPANSION YES)
set(DOXYGEN_PREDEFINED "SVS_UNUSED(x)=x")

# Path to example code
set(DOXYGEN_EXAMPLE_PATH "${EXAMPLE_CPP_DIR}/")
set(DOXYGEN_EXAMPLE_PATTERNS "*.cpp")
set(DOXYGEN_EXAMPLE_RECURSIVE NO)

# Only extract documented files.
set(DOXYGEN_EXTRACT_ALL NO)
set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doxygen")

# Hide undocumented structs.
set(DOXYGEN_HIDE_UNDOC_MEMBERS YES)
set(DOXYGEN_HIDE_UNDOC_CLASSES YES)
set(DOXYGEN_HIDE_FRIEND_COMPOUNDS YES)
set(DOXYGEN_HIDE_IN_BODY_DOCS YES)

# Get file-paths to resolve correctly in Doxygen.
set(DOXYGEN_FORCE_LOCAL_INCLUDES NO)
set(DOXYGEN_STRIP_FROM_INC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../include")

# In the few places where we use inheritance, it's usually because we want to include
# implementations in a partial base class.
#
# Setting this to `true` lets us inherit documentation from the base class.
set(DOXYGEN_INLINE_INHERITED_MEMB YES)

set(DOXYGEN_TARGET "docs")

file(GLOB_RECURSE DOXYGEN_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/include/svs/*"
    "${EXAMPLE_CPP_DIR}/*"
)

message("Doxygen Files: ${DOXYGEN_SOURCE_FILES}")

# defined implicitly in doxygen_add_docs
set(DOXYGEN_STAMP_FILE "${CMAKE_CURRENT_BINARY_DIR}/${DOXYGEN_TARGET}.stamp")

doxygen_add_docs(
    docs ALL
    ${DOXYGEN_SOURCE_FILES}
    USE_STAMP_FILE
    COMMENT "Generating Documentation"
)

# Setup Sphinx
# Make sure we copy python examples into the correct directory.
find_package(Sphinx REQUIRED)
set(SPHINX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}")
set(SPHINX_BUILD "${CMAKE_CURRENT_BINARY_DIR}/sphinx")

# Sphinx Executable flags:
# * -b html : Built an HTML website.
# * -W : Turn warnings into errors.
add_custom_target(
    Sphinx ALL
    COMMAND
    ${SPHINX_EXECUTABLE} -b html
        -Dbreathe_projects.SVS="${DOXYGEN_OUTPUT_DIRECTORY}/xml"
        ${SPHINX_SOURCE} ${SPHINX_BUILD}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
        ${DOXYGEN_STAMP_FILE}
    COMMENT "Generating documentation with Sphinx"
)

