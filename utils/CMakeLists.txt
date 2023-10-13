function(create_utility exe file)
    add_executable(${exe} ${file})
    target_include_directories(
        ${exe}
        PRIVATE ${CMAKE_CURRENT_LIST_DIR}
    )
    # Get header directory and C++ 20
    target_link_libraries(${exe} PRIVATE ${SVS_LIB})

    # Get common compiler options with the unit tests.
    target_link_libraries(${exe} PRIVATE svs_compile_options svs_native_options)

    # Link with third-party executables.
    target_link_libraries(${exe} PRIVATE fmt::fmt)

    # Optionally apply clang-tidy to the executable
    if (SVS_EXPERIMENTAL_CLANG_TIDY AND CLANG_TIDY_COMMAND)
        set_target_properties(
            ${exe}
            PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}"
        )
    endif()
endfunction()

create_utility(graph_stat graph_stat.cpp)

# Legacy conversion routines.
create_utility(convert_legacy convert_legacy.cpp)
create_utility(assemble_vamana assemble_vamana.cpp)
create_utility(generate_vamana_config generate_vamana_config.cpp)

create_utility(build_index build_index.cpp)
create_utility(compute_recall compute_recall.cpp)
create_utility(convert_data_to_float16 convert_data_to_float16.cpp)
create_utility(search_index search_index.cpp)
create_utility(consolidate characterization/consolidate.cpp)
# create_utility(mutable characterization/mutable.cpp)

# # Quantization
# create_utility(build_quantized quantization/build_quantized.cpp)
# create_utility(dynamic_lvq quantization/dynamic_lvq.cpp)
# create_utility(clustered_lvq quantization/clustered_lvq.cpp)

# # Benchmark
# create_utility(benchmark_index_build benchmarks/index_build.cpp)
