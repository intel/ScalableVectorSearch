/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
 */
// svs
#include "svs/core/io.h"
#include "svs/lib/dispatcher.h"
#include "svs/lib/file.h"
#include "svs/lib/file_iterator.h"
#include "svs/lib/timing.h"

// svsmain
#include "svsmain.h"

// format
#include "fmt/core.h"

// stl
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

const size_t LEGACY_HEADER_SIZE_BYTES = 64;

/////
///// Graph Conversion
/////

void convert_graph(const std::vector<std::string>& args) {
    const auto& source_path = args.at(2);
    const auto& dest_path = args.at(3);

    // Construct a native file writer and hack together a reader for the old file format.
    auto src = svs::lib::open_read(source_path);
    auto num_vectors = svs::lib::read_binary<size_t>(src);
    auto vector_dim = svs::lib::read_binary<size_t>(src);
    fmt::print("Converting {} vertices with max degree {}\n", num_vectors, vector_dim);
    src.seekg(LEGACY_HEADER_SIZE_BYTES, std::ifstream::beg);

    auto tic = svs::lib::now();
    auto reader = svs::lib::heterogeneous_iterator(
        src,
        num_vectors,
        svs::lib::ValueReader<size_t>(),
        svs::lib::VectorReader<uint32_t>(vector_dim)
    );

    auto writer = svs::io::NativeFile(dest_path).writer(vector_dim + 1);
    auto buffer = std::vector<uint32_t>(vector_dim + 1);
    auto end = svs::lib::HeterogeneousFileEnd{};
    while (reader != end) {
        auto [count, neighbors] = *reader;
        buffer.at(0) = count;
        std::copy(neighbors.begin(), neighbors.end(), buffer.begin() + 1);
        writer << svs::lib::as_const_span(buffer);
        // Increment iterator
        ++reader;
    }
    fmt::print("Conversion took {} seconds\n", svs::lib::time_difference(tic));
}

/////
///// Data Conversion
/////

template <typename T>
void convert_data_impl(
    svs::lib::Type<T> SVS_UNUSED(dispatch),
    const std::filesystem::path& src_path,
    const std::filesystem::path& dst_path
) {
    auto src = svs::lib::open_read(src_path);
    auto num_vectors = svs::lib::read_binary<size_t>(src);
    auto vector_dim = svs::lib::read_binary<size_t>(src);
    fmt::print("Converting {} vectors with dimension {}\n", num_vectors, vector_dim);
    src.seekg(LEGACY_HEADER_SIZE_BYTES, std::ifstream::beg);

    auto tic = svs::lib::now();
    auto reader = svs::lib::heterogeneous_iterator(
        src, num_vectors, svs::lib::VectorReader<T>(vector_dim)
    );

    auto writer = svs::io::NativeFile(dst_path).writer(vector_dim);
    auto buffer = std::vector<T>(vector_dim);
    auto end = svs::lib::HeterogeneousFileEnd{};
    while (reader != end) {
        auto data = *reader;
        std::copy(data.begin(), data.end(), buffer.begin());
        writer << svs::lib::as_const_span(buffer);
        // Increment iterator
        ++reader;
    }
    fmt::print("Conversion took {} seconds\n", svs::lib::time_difference(tic));
}

using Dispatcher = svs::lib::Dispatcher<
    void,
    svs::DataType,
    const std::filesystem::path&,
    const std::filesystem::path&>;

void convert_data(const std::vector<std::string>& args) {
    const auto& eltype = svs::parse_datatype(args.at(2));
    const auto& src_path = args.at(3);
    const auto& dst_path = args.at(4);

    // Build a dispatch table.
    auto dispatch = Dispatcher();
    constexpr auto types = svs::lib::Types<float, svs::Float16, uint8_t, int8_t>();
    svs::lib::for_each_type(types, [&]<typename T>(svs::lib::Type<T> SVS_UNUSED(type)) {
        dispatch.register_target(&convert_data_impl<T>);
    });

    dispatch.invoke(eltype, src_path, dst_path);
}

/////
///// Main
/////

constexpr std::string_view expected_nargs = "3 or 4";
constexpr std::string_view help = R"(
Usage: convert_legacy kind [element_type] source dest

Convert legacy data and graph files into the new Version 1.0 format.
Arguments:
    kind         - The kind of file to convert. Can be either "graph" or "data".
    element_type - Required if "kind == data", describes the vector element type of the
                   corresponding dataset. Possible values:
                   "float32", "float16", "uint8", "int8".
    source       - The path to the original file on disk.
    dest         - The path where the new file will be generated.
)";

void print_help() { fmt::print(help); }

int svs_main(std::vector<std::string> args) {
    auto nargs = args.size();
    switch (nargs) {
        case 4: {
            const auto& kind = args.at(1);
            if (kind != "graph") {
                print_help();
                return 1;
            }
            convert_graph(args);
            break;
        }
        case 5: {
            const auto& kind = args.at(1);
            if (kind != "data") {
                print_help();
                return 1;
            }
            convert_data(args);
            break;
        }
        default: {
            fmt::print(
                "Unknown number of args. Got {}, expected {}.", nargs - 1, expected_nargs
            );
            print_help();
            return 1;
        }
    }
    return 0;
}

SVS_DEFINE_MAIN();
