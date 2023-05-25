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

// stdlib
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>

// svs
#include "svs/lib/file_iterator.h"
#include "svs/lib/readwrite.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// tests
#include "tests/utils/generators.h"

CATCH_TEST_CASE("Testing Readers", "[lib][file_iterator]") {
    auto stream = std::stringstream{};
    CATCH_SECTION("Vector Reader") {
        auto v = std::vector<double>{};
        auto generator = svs_test::make_generator<double>(-100, 100);
        svs_test::populate(v, generator, 1000);
        svs::lib::write_binary(stream, v);
        stream.seekg(0, std::stringstream::beg);

        // Construct a VectorReader.
        auto reader = svs::lib::VectorReader<double>(v.size());
        CATCH_REQUIRE(reader.size() == v.size());
        // Default data should not be equal.
        auto data = reader.data();
        CATCH_REQUIRE(std::equal(data.begin(), data.end(), v.begin()) == false);
        reader.read(stream);
        data = reader.data();
        CATCH_REQUIRE(std::equal(data.begin(), data.end(), v.begin()) == true);

        // Now - do two writes to ensure that the underlying buffer is incremented
        // properly.
        svs_test::populate(v, generator, 100);
        auto u = std::vector<double>{};
        svs_test::populate(u, generator, 500);

        stream.seekp(0, std::stringstream::beg);
        svs::lib::write_binary(stream, v);
        svs::lib::write_binary(stream, u);
        stream.seekg(0, std::stringstream::beg);

        // Read the chunk added by `v`.
        reader.resize(v.size());
        reader.read(stream);
        data = reader.data();
        CATCH_REQUIRE(data.size() == v.size());
        CATCH_REQUIRE(std::equal(data.begin(), data.end(), v.begin()));

        // Read the chunk added by `u`.
        reader.resize(u.size());
        reader.read(stream);
        data = reader.data();
        CATCH_REQUIRE(data.size() == u.size());
        CATCH_REQUIRE(std::equal(data.begin(), data.end(), u.begin()));
    }

    CATCH_SECTION("Value Reader") {
        svs::lib::write_binary(stream, size_t{100});
        svs::lib::write_binary(stream, float{-100});
        svs::lib::write_binary(stream, uint8_t{0x5});
        svs::lib::write_binary(stream, size_t{1234});

        auto size_reader = svs::lib::ValueReader<size_t>();
        auto float_reader = svs::lib::ValueReader<float>();
        auto uint8_reader = svs::lib::ValueReader<uint8_t>();

        size_reader.read(stream);
        CATCH_REQUIRE(size_reader.data() == 100);
        float_reader.read(stream);
        CATCH_REQUIRE(float_reader.data() == -100);
        uint8_reader.read(stream);
        CATCH_REQUIRE(uint8_reader.data() == 0x5);

        // Reread with the `size_reader` to test replacement.
        size_reader.read(stream);
        CATCH_REQUIRE(size_reader.data() == 1234);
    }

    CATCH_SECTION("HeterogeneousFileIterator") {
        // Create a temporary file.
        CATCH_REQUIRE(svs_test::prepare_temp_directory());
        std::string path = svs_test::temp_directory() / "temp.bin";
        auto ostream = std::ofstream{path, std::ofstream::out | std::ofstream::binary};

        const size_t dataset_size = 100;
        const size_t vector_length = 111;
        // Create a dummy dataset composed of a mix of data types.
        auto vectors = std::vector<std::vector<float>>(dataset_size);
        auto meta_size_t = std::vector<size_t>(dataset_size);
        auto meta_uint8_t = std::vector<uint8_t>(dataset_size);

        // Populate the dataset.
        auto float_generator = svs_test::make_generator<float>(-1000, 1000);
        auto uint8_t_generator = svs_test::make_generator<uint8_t>(0, 100);
        for (auto& i : vectors) {
            svs_test::populate(i, float_generator, vector_length);
        }
        svs_test::populate(
            meta_size_t, svs_test::make_generator<size_t>(0, 10000), dataset_size
        );
        svs_test::populate(
            meta_uint8_t, svs_test::make_generator<uint8_t>(0, 100), dataset_size
        );

        // Write dataset to stringstream.
        for (size_t i = 0; i < dataset_size; ++i) {
            svs::lib::write_binary(ostream, meta_size_t[i]);
            svs::lib::write_binary(ostream, meta_uint8_t[i]);
            svs::lib::write_binary(ostream, vectors[i]);
        }
        ostream.flush();

        // Now that everything has been written into the buffer, we can construct a
        // `HeterogeneousFileIterator` to read back from the file.
        auto istream = std::ifstream{path, std::ifstream::in | std::ifstream::binary};
        auto reader = svs::lib::VectorReader<float>(vector_length);

        // Here, `original_data_pointer` will point to the beginning of the memory
        // allocated for the buffer stored inside the `VectorBuffer`.
        //
        // Throughout the tuple handling logic, we want to make sure that the data returned
        // by the iterator is to the **same** underlying buffer.
        //
        // That is, there should be no copies happening (for efficiency purposes).
        auto original_data_pointer = reader.data().data();
        auto iter = svs::lib::heterogeneous_iterator(
            istream,
            dataset_size,
            svs::lib::ValueReader<size_t>(),
            svs::lib::ValueReader<uint8_t>(),
            std::move(reader)
        );
        auto count = 0;
        while (iter != svs::lib::HeterogeneousFileEnd{}) {
            auto tup = *iter;
            CATCH_REQUIRE(std::get<0>(tup) == meta_size_t[count]);
            CATCH_REQUIRE(std::get<1>(tup) == meta_uint8_t[count]);
            auto& v = std::get<2>(tup);
            CATCH_REQUIRE(v.data() == original_data_pointer);
            CATCH_REQUIRE(v.size() == vectors[count].size());
            CATCH_REQUIRE(v.size() == vector_length);

            CATCH_REQUIRE(std::equal(v.begin(), v.end(), vectors[count].begin()));
            // Make sure that we have the same data pointer throughout (no copies).
            CATCH_REQUIRE(v.data() == original_data_pointer);

            // Increment indexing.
            ++count;
            ++iter;
        }
        CATCH_REQUIRE(count == dataset_size);
    }
}
