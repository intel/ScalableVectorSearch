if(SVS_BUILD_TESTS OR SVS_TEST_EXAMPLES)
    # N.B.: Including `CTest` automatically calls `enable_testing()`.
    # Without this include, `enable_testing()` is never called and thus `add_test()`
    # has no effect.
    include(CTest)
endif()

# Create a simple [1] example executable called `exe` from the file `file`.
# Also creates a test executable called `test` if `SVS_ENABLE_TESTS` is enabled.
#
# [1] A simple executable is one that takes no commandline arguments.
function(create_simple_example exe test file)
    add_executable(${exe} ${file})
    target_include_directories(${exe} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    # Link to our library
    target_link_libraries(${exe} ${SVS_LIB} svs_compile_options svs_native_options)
    # Create a test.
    # No-op if the `include(CTest)` line above is not executed.
    add_test(${test} ${exe})
endfunction()

# Examples
create_simple_example(hello_world test_hello_world hello_world.cpp)
create_simple_example(saveload test_saveload saveload.cpp)
create_simple_example(types test_types types.cpp)

## More complicated examples involving more extensive setup.

#####
##### Vamana Test
#####

# Create a setup fixture to generate the test dataset from pysvs
add_test(
    NAME generate_dataset
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/test_dataset.py
)
set_tests_properties(generate_dataset PROPERTIES FIXTURES_SETUP gen_dataset)

# The vamana test executable.
# Have it depend on the `gen_dataset` fixture so we can generated the dataset before
# executing the test.
add_executable(vamana vamana.cpp)
target_include_directories(vamana PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(vamana ${SVS_LIB} svs_compile_options svs_native_options)
add_test(
    NAME test_vamana
    COMMAND
        vamana
        example_data_vamana/data.fvecs
        example_data_vamana/queries.fvecs
        example_data_vamana/groundtruth.ivecs
)
set_tests_properties(test_vamana PROPERTIES FIXTURES_REQUIRED gen_dataset)

#####
##### Dispatcher
#####

add_executable(dispatcher dispatcher.cpp)
target_include_directories(dispatcher PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(dispatcher ${SVS_LIB} svs_compile_options svs_native_options)

# Here we go.
add_test(
    NAME test_dispatcher_1
    COMMAND dispatcher float16 float16 128 false hello
)
set_property(TEST test_dispatcher_1
    PROPERTY
        PASS_REGULAR_EXPRESSION "Generic: float16, float16, dynamic with arg \"hello\""
)

add_test(
    NAME test_dispatcher_2
    COMMAND dispatcher float16 float16 128 false true
)
set_property(TEST test_dispatcher_2
    PROPERTY
        PASS_REGULAR_EXPRESSION "Generic: float16, float16, dynamic with arg \"boolean true\""
)

add_test(
    NAME test_dispatcher_3
    COMMAND dispatcher float16 float16 128 true true
)
set_property(TEST test_dispatcher_3
    PROPERTY
    PASS_REGULAR_EXPRESSION "ANNException"
)

add_test(
    NAME test_dispatcher_4
    COMMAND dispatcher uint32 uint8 128 true hello
)
set_property(TEST test_dispatcher_4
    PROPERTY
    PASS_REGULAR_EXPRESSION "Specialized with string: uint32, uint8, 128 with arg \"hello\""
)

add_test(
    NAME test_dispatcher_5
    COMMAND dispatcher float32 float32 100 false false
)
set_property(TEST test_dispatcher_5
    PROPERTY
    PASS_REGULAR_EXPRESSION "Specialized with flag: float32, float32, dynamic with arg \"false\""
)
