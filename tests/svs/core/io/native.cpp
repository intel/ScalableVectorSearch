/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

// test headers
#include "svs/core/io/native.h"
#include "svs/core/io/vecs.h"

// tests
#include "tests/utils/schemas.h"
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

#include "catch2/catch_test_macros.hpp"

// stl
#include <iostream>
#include <numeric>
#include <span>
#include <vector>

CATCH_TEST_CASE("Schemas", "[core][io][schema]") {
    using FileSchema = svs::io::FileSchema;
    CATCH_SECTION("Names") {
        CATCH_REQUIRE(svs::io::name<FileSchema::Vtest>() == "Vtest");
        CATCH_REQUIRE(svs::io::name<FileSchema::V1>() == "V1");

        CATCH_REQUIRE(svs::io::name(FileSchema::Vtest) == "Vtest");
        CATCH_REQUIRE(svs::io::name(FileSchema::V1) == "V1");

        CATCH_REQUIRE(svs::io::parse_schema("Vtest") == FileSchema::Vtest);
        CATCH_REQUIRE(svs::io::parse_schema("V1") == FileSchema::V1);
        CATCH_REQUIRE_THROWS_AS(svs::io::parse_schema("Vnone"), svs::ANNException);
    }

    CATCH_SECTION("Magic") {
        namespace io = svs::io;
        CATCH_REQUIRE(io::from_magic_number(io::vtest::magic_number) == FileSchema::Vtest);
        CATCH_REQUIRE(io::from_magic_number(io::v1::magic_number) == FileSchema::V1);
        CATCH_REQUIRE(!io::from_magic_number(0).has_value());

        CATCH_REQUIRE(
            io::get_magic_number(test_schemas::test_vtest_file()) == io::vtest::magic_number
        );
        CATCH_REQUIRE(
            io::get_magic_number(test_schemas::test_v1_file()) == io::v1::magic_number
        );
    }

    CATCH_SECTION("Classification") {
        namespace io = svs::io;
        CATCH_REQUIRE(io::classify(test_schemas::test_vtest_file()) == FileSchema::Vtest);
        CATCH_REQUIRE(io::classify(test_schemas::test_v1_file()) == FileSchema::V1);
    }
}

CATCH_TEST_CASE("Testing Native Reader Iterator", "[core][io]") {
    CATCH_REQUIRE(svs_test::prepare_temp_directory());
    auto vecs_file = test_dataset::reference_vecs_file();
    auto native_file = test_dataset::reference_svs_file();

    // Load the reference contents of the file and pre-compute some statistics
    // about the reference contents.
    auto reference = test_dataset::reference_file_contents();
    auto reference_ndims = reference.at(0).size();
    auto reference_nvectors = reference.size();
    CATCH_REQUIRE(reference_ndims != reference_nvectors);

    CATCH_SECTION("Loading") {
        auto eltype = svs::lib::meta::Type<float>();
        auto file = svs::io::v1::NativeFile{native_file};
        auto [nvectors, ndims] = file.get_dims();
        CATCH_REQUIRE(ndims == reference_ndims);
        CATCH_REQUIRE(nvectors == reference_nvectors);

        auto loader = file.reader(eltype, 1);
        CATCH_REQUIRE(loader.ndims() == reference_ndims);
        auto v = std::vector<float>{};
        for (auto i : loader) {
            v.insert(v.end(), i.begin(), i.end());
        }
        CATCH_REQUIRE(v.size() == reference.at(0).size());
        CATCH_REQUIRE(std::equal(v.begin(), v.end(), reference.at(0).begin()));

        // Read entire file
        loader.resize(nvectors);
        auto loader_it = loader.begin();
        auto loader_end = loader.end();
        for (size_t i = 0, imax = reference.size(); i < imax; ++i) {
            v.clear();
            const auto& this_reference = reference.at(i);
            CATCH_REQUIRE(loader_it != loader_end);
            const auto& slice = *loader_it;
            CATCH_REQUIRE(slice.size() == this_reference.size());
            CATCH_REQUIRE(std::equal(slice.begin(), slice.end(), this_reference.begin()));
            ++loader_it;
        }
        CATCH_REQUIRE(!(loader_it != loader_end));
    }

    CATCH_SECTION("Compare with Vecs") {
        auto eltype = svs::lib::meta::Type<float>();
        auto vecs_loader = svs::io::vecs::VecsFile<float>(vecs_file).reader(eltype);
        auto native_loader = svs::io::v1::NativeFile(native_file).reader(eltype);

        auto vecs_data = std::vector<float>{};
        auto native_data = std::vector<float>{};
        for (auto i : vecs_loader) {
            vecs_data.insert(vecs_data.end(), i.begin(), i.end());
        }
        for (auto i : native_loader) {
            native_data.insert(native_data.end(), i.begin(), i.end());
        }
        CATCH_REQUIRE(vecs_data.size() == native_data.size());
        CATCH_REQUIRE(
            std::equal(vecs_data.begin(), vecs_data.end(), native_data.begin()) == true
        );
    }

    CATCH_SECTION("Writing") {
        auto eltype = svs::lib::meta::Type<float>();
        auto file = svs::io::v1::NativeFile(native_file);
        auto uuid = file.uuid();
        auto reader = file.reader(eltype);

        CATCH_REQUIRE(reader.ndims() == reference_ndims);
        CATCH_REQUIRE(reader.nvectors() == reference_nvectors);
        std::string output_file = svs_test::temp_directory() / "output.svs";

        // Load reference data.
        auto reference = std::vector<float>{};
        for (auto i : reader) {
            reference.insert(reference.end(), i.begin(), i.end());
        }

        CATCH_SECTION("Simple Writing") {
            auto writer = svs::io::v1::NativeFile(output_file).writer(reader.ndims(), uuid);
            for (auto i : reader) {
                writer << i;
            }
            writer.writeheader();
            writer.flush();
            CATCH_REQUIRE(svs_test::compare_files(native_file, output_file) == true);
        }
    }
}

// Now, try graph loading and saving.
CATCH_TEST_CASE("Testing Native Reader Graph IO", "[core][io]") {
    CATCH_REQUIRE(svs_test::prepare_temp_directory());
    auto graph_file = test_dataset::graph_file();
    CATCH_SECTION("Basic metadata info") {
        auto eltype = svs::lib::meta::Type<uint32_t>();
        auto file = svs::io::v1::NativeFile{graph_file};
        auto reader = file.reader(eltype);
        CATCH_REQUIRE(reader.ndims() == test_dataset::GRAPH_MAX_DEGREE + 1);
        CATCH_REQUIRE(reader.nvectors() == test_dataset::VECTORS_IN_DATA_SET);
        // Manually extract the first few sizes to verify reading is working properly.
        size_t i = 0;
        auto expected = test_dataset::expected_out_neighbors();
        for (const auto& span : reader) {
            CATCH_REQUIRE(span.front() == expected.at(i));
            CATCH_REQUIRE(span.size() == test_dataset::GRAPH_MAX_DEGREE + 1);
            auto neighbors = span.subspan(1, span.front());
            for (auto& n : neighbors) {
                CATCH_REQUIRE(n < test_dataset::VECTORS_IN_DATA_SET);
            }
            ++i;
            if (i == expected.size()) {
                break;
            }
        }
    }
}

namespace {
template <typename T, typename U> void copyto(std::span<T> dst, const std::vector<U>& src) {
    CATCH_REQUIRE(dst.size() == src.size());
    std::copy(src.begin(), src.end(), dst.begin());
}
} // namespace

CATCH_TEST_CASE("File Detection", "[core][file_detection]") {
    namespace io = svs::io;
    CATCH_SECTION("File Type") {
        CATCH_REQUIRE(std::is_same_v<
                      io::file_type_t<io::FileSchema::Vtest>,
                      io::vtest::NativeFile>);
        CATCH_REQUIRE(std::is_same_v<
                      io::file_type_t<io::FileSchema::V1>,
                      svs::io::v1::NativeFile>);
    }

    CATCH_SECTION("Visit File Type") {
        auto path = std::filesystem::path("a path");
        io::visit_file_type(io::FileSchema::Vtest, path, [&](const auto& file) {
            using T = std::decay_t<decltype(file)>;
            CATCH_REQUIRE(std::is_same_v<T, svs::io::vtest::NativeFile>);
            CATCH_REQUIRE(file.get_path() == path);
        });

        io::visit_file_type(io::FileSchema::V1, path, [&](const auto& file) {
            using T = std::decay_t<decltype(file)>;
            CATCH_REQUIRE(std::is_same_v<T, svs::io::v1::NativeFile>);
            CATCH_REQUIRE(file.get_path() == path);
        });
    }

    CATCH_SECTION("Get UUID") {
        // Detection of Vtest
        auto uuid = io::get_uuid(test_schemas::test_vtest_file());
        CATCH_REQUIRE(uuid.value() == test_schemas::vtest_uuid());
        CATCH_REQUIRE(uuid.value() != test_schemas::v1_uuid());

        // Detection of V1
        uuid = io::get_uuid(test_schemas::test_v1_file());
        CATCH_REQUIRE(uuid.has_value());
        CATCH_REQUIRE(uuid.value() == test_schemas::v1_uuid());
        CATCH_REQUIRE(uuid.value() != test_schemas::vtest_uuid());
    }
}
