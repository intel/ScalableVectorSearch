/*
 * Copyright 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "svs/core/io/memstream.h"

#include "tests/utils/utils.h"

#include "catch2/catch_test_macros.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

namespace {
std::filesystem::path write_file(const std::string& name, const std::string& contents) {
    auto path = svs_test::prepare_temp_directory_v2() / name;
    auto out = std::ofstream(path, std::ios::binary);
    out << contents;
    out.close();
    return path;
}

std::filesystem::path create_empty_file(const std::string& name) {
    auto path = svs_test::prepare_temp_directory_v2() / name;
    std::ofstream(path, std::ios::binary).close();
    return path;
}
} // namespace

CATCH_TEST_CASE("mmstreambuf reads and seeks", "[core][io][mmap]") {
    auto path = write_file("mmstream_data.bin", "0123456789");

    auto buf = svs::io::mmstreambuf{};
    buf.open(path);
    CATCH_REQUIRE(buf.is_open());
    CATCH_REQUIRE(buf.size() == 10);

    CATCH_REQUIRE(buf.sgetc() == '0');
    CATCH_REQUIRE(buf.sbumpc() == '0');
    CATCH_REQUIRE(buf.sgetc() == '1');

    CATCH_REQUIRE(buf.pubseekoff(4, std::ios_base::beg, std::ios_base::in) == 4);
    CATCH_REQUIRE(buf.sgetc() == '4');

    CATCH_REQUIRE(buf.pubseekoff(-1, std::ios_base::cur, std::ios_base::in) == 3);
    CATCH_REQUIRE(buf.sgetc() == '3');

    CATCH_REQUIRE(buf.pubseekoff(-1, std::ios_base::beg, std::ios_base::in) == -1);
    CATCH_REQUIRE(buf.pubseekoff(1, std::ios_base::end, std::ios_base::in) == -1);

    CATCH_REQUIRE(buf.pubseekpos(9, std::ios_base::in) == 9);
    CATCH_REQUIRE(buf.sgetc() == '9');

    CATCH_REQUIRE(buf.pubseekpos(10, std::ios_base::in) == 10);
    CATCH_REQUIRE(buf.sgetc() == std::char_traits<char>::eof());

    buf.close();
    CATCH_REQUIRE(!buf.is_open());
    CATCH_REQUIRE(buf.size() == 0);
}

CATCH_TEST_CASE("mmstreambuf handles empty files", "[core][io][mmap]") {
    auto path = create_empty_file("mmstream_empty.bin");

    auto buf = svs::io::mmstreambuf{};
    CATCH_REQUIRE_THROWS_AS(buf.open(path), svs::lib::ANNException);
    CATCH_REQUIRE(!buf.is_open());
    CATCH_REQUIRE(buf.size() == 0);
    CATCH_REQUIRE(buf.sgetc() == std::char_traits<char>::eof());
}

CATCH_TEST_CASE("mmstreambuf supports move operations", "[core][io][mmap]") {
    auto path = write_file("mmstream_move.bin", "abcdef");

    auto source = svs::io::mmstreambuf{};
    source.open(path);
    CATCH_REQUIRE(source.pubseekpos(2, std::ios_base::in) == 2);

    auto moved = svs::io::mmstreambuf(std::move(source));
    CATCH_REQUIRE(moved.is_open());
    CATCH_REQUIRE(!source.is_open());
    CATCH_REQUIRE(moved.sgetc() == 'c');

    auto assigned = svs::io::mmstreambuf{};
    assigned = std::move(moved);
    CATCH_REQUIRE(assigned.is_open());
    CATCH_REQUIRE(!moved.is_open());
    CATCH_REQUIRE(assigned.sgetc() == 'c');
}

CATCH_TEST_CASE("mmstream provides istream interface", "[core][io][mmap]") {
    auto path = write_file("mmstream_stream.bin", "line1\nline2\n");

    auto stream = svs::io::mmstream(path);
    CATCH_REQUIRE(stream.is_open());
    CATCH_REQUIRE(stream.size() == 12);

    auto line = std::string{};
    std::getline(stream, line);
    CATCH_REQUIRE(line == "line1");

    stream.seekg(0, std::ios_base::beg);
    std::getline(stream, line);
    CATCH_REQUIRE(line == "line1");

    stream.seekg(6, std::ios_base::beg);
    std::getline(stream, line);
    CATCH_REQUIRE(line == "line2");

    stream.close();
    CATCH_REQUIRE(!stream.is_open());
    CATCH_REQUIRE(stream.eof());
}

CATCH_TEST_CASE("current_ptr pointer semantics", "[core][io][mmap]") {
    auto path = write_file("mmstream_ptrs.bin", "ABCDE");
    auto stream = svs::io::mmstream(path);

    auto* b = svs::io::current_ptr<char>(stream);
    CATCH_REQUIRE(b != nullptr);
    CATCH_REQUIRE(*b == 'A');

    for (std::size_t i = 0; i < stream.size(); ++i) {
        CATCH_REQUIRE(
            svs::io::current_ptr<char>(stream) == b + static_cast<std::ptrdiff_t>(i)
        );
        stream.ignore(1);
    }

    stream.seekg(0, std::ios_base::beg);
    CATCH_REQUIRE(svs::io::current_ptr<char>(stream) == b);

    stream.seekg(3, std::ios_base::beg);
    CATCH_REQUIRE(svs::io::current_ptr<char>(stream) == b + 3);

    stream.close();
    CATCH_REQUIRE(svs::io::current_ptr<char>(stream) == nullptr);
}

CATCH_TEST_CASE("mmstream open throws on missing file", "[core][io][mmap]") {
    auto missing = svs_test::prepare_temp_directory_v2() / "mmstream_missing.bin";

    auto stream = svs::io::mmstream{};
    CATCH_REQUIRE_THROWS_AS(stream.open(missing), svs::lib::ANNException);
}

CATCH_TEST_CASE("is_memory_stream", "[core][io][mmap]") {
    // mmstream is an in-memory stream.
    auto path = write_file("mmstream_inmem.bin", "hello");
    auto mm = svs::io::mmstream(path);
    CATCH_REQUIRE(svs::io::is_memory_stream(mm));

    // std::istringstream is an in-memory stream.
    auto iss = std::istringstream("world");
    CATCH_REQUIRE(svs::io::is_memory_stream(iss));

    // std::stringstream is an in-memory stream.
    auto ss = std::stringstream("test");
    CATCH_REQUIRE(svs::io::is_memory_stream(ss));

    // svs::io::spanstream is an in-memory stream.
    char buffer[] = "span";
    auto span = svs::io::ispanstream(std::span<char>{buffer});
    CATCH_REQUIRE(svs::io::is_memory_stream(span));

    // std::ifstream is NOT an in-memory stream.
    auto ifs = std::ifstream(path);
    CATCH_REQUIRE(!svs::io::is_memory_stream(ifs));
}

CATCH_TEST_CASE("current_ptr", "[core][io][mmap]") {
    // ---- mmstream ----
    // Write 3 floats to a binary file and open it as an mmstream.
    const std::array<float, 3> data = {1.0f, 2.0f, 3.0f};
    auto path = svs_test::prepare_temp_directory_v2() / "memstream_ptr.bin";
    {
        auto out = std::ofstream(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()), sizeof(data));
    }

    auto mm = svs::io::mmstream(path);
    auto* p0 = svs::io::current_ptr<float>(mm);
    CATCH_REQUIRE(p0 != nullptr);
    CATCH_REQUIRE(*p0 == data[0]);

    // Reading one float worth of bytes advances current() by sizeof(float).
    mm.ignore(static_cast<std::streamsize>(sizeof(float)));
    auto* p1 = svs::io::current_ptr<float>(mm);
    CATCH_REQUIRE(p1 == p0 + 1);
    CATCH_REQUIRE(*p1 == data[1]);

    // Seeking back to the start returns the same base pointer.
    mm.seekg(0, std::ios_base::beg);
    CATCH_REQUIRE(svs::io::current_ptr<float>(mm) == p0);

    // After close the stream is not in-memory anymore: returns nullptr.
    mm.close();
    CATCH_REQUIRE(svs::io::current_ptr<float>(mm) == nullptr);

    // ---- std::istringstream ----
    // Build a string with known byte content, then read it back as chars.
    const std::string text = "ABCDE";
    auto iss = std::istringstream(text);
    auto* cp0 = svs::io::current_ptr<char>(iss);
    CATCH_REQUIRE(cp0 != nullptr);
    CATCH_REQUIRE(*cp0 == 'A');

    iss.ignore(2);
    auto* cp1 = svs::io::current_ptr<char>(iss);
    CATCH_REQUIRE(cp1 == cp0 + 2);
    CATCH_REQUIRE(*cp1 == 'C');

    // ---- std::ifstream — not in-memory: must return nullptr ----
    {
        auto ifs = std::ifstream(path, std::ios::binary);
        CATCH_REQUIRE(svs::io::current_ptr<float>(ifs) == nullptr);
    }

    // ---- empty std::istringstream: in-memory but empty, must return nullptr ----
    {
        auto empty_iss = std::istringstream("");
        CATCH_REQUIRE(svs::io::is_memory_stream(empty_iss));
        CATCH_REQUIRE(svs::io::current_ptr<char>(empty_iss) == nullptr);
    }
}

CATCH_TEST_CASE("spanstream current_ptr", "[core][io][mmap]") {
    char text[] =
        "Hello, world!"; // Note: not a string literal, so we can take its address.
    auto iss = svs::io::ispanstream(std::span<char>{text});
    auto* base_ptr = svs::io::current_ptr<char>(iss);
    CATCH_REQUIRE(base_ptr != nullptr);
    CATCH_REQUIRE(*base_ptr == 'H');

    for (std::size_t i = 0; i < std::strlen(text); ++i) {
        auto* current = svs::io::current_ptr<char>(iss);
        auto* expected = text + i;
        auto match = (current == expected) && (*current == text[i]);
        CATCH_REQUIRE(match);
        // CATCH_REQUIRE(svs::io::current_ptr<char>(iss) == text + i);
        iss.ignore(1);
    }

    // After reading all characters, current_ptr should point to the null terminator.
    CATCH_REQUIRE(svs::io::current_ptr<char>(iss) == text + std::strlen(text));
    CATCH_REQUIRE(*svs::io::current_ptr<char>(iss) == '\0');
}

CATCH_TEST_CASE("ispanstream span() getter and setter", "[core][io][mmap]") {
    char text1[] = "First";
    char text2[] = "Second";

    auto stream = svs::io::ispanstream(std::span<char>{text1});

    // Test getter
    auto s1 = stream.rdbuf()->span();
    CATCH_REQUIRE(s1.data() == text1);
    CATCH_REQUIRE(s1.size() == 6);

    // Test setter with new span
    stream.rdbuf()->span(std::span<char>{text2});
    auto s2 = stream.rdbuf()->span();
    CATCH_REQUIRE(s2.data() == text2);
    CATCH_REQUIRE(s2.size() == 7);

    // Verify position resets to beginning
    CATCH_REQUIRE(svs::io::current_ptr<char>(stream) == text2);
    CATCH_REQUIRE(*svs::io::current_ptr<char>(stream) == 'S');
}

CATCH_TEST_CASE("ispanstream with empty span", "[core][io][mmap]") {
    std::span<char> empty;
    auto stream = svs::io::ispanstream(empty);

    CATCH_REQUIRE(stream.rdbuf()->span().empty());
    CATCH_REQUIRE(svs::io::is_memory_stream(stream));
    CATCH_REQUIRE(svs::io::current_ptr<char>(stream) == nullptr);
    CATCH_REQUIRE(stream.rdbuf()->span().size() == 0);

    // Setting non-empty span should work
    char text[] = "data";
    stream.rdbuf()->span(std::span<char>{text});
    CATCH_REQUIRE(!stream.rdbuf()->span().empty());
    CATCH_REQUIRE(svs::io::current_ptr<char>(stream) == text);
}

CATCH_TEST_CASE("MemoryStreamAllocator", "[core][io][mmap]") {
    // Create a buffer with float data
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    auto run_allocator_checks = [&](std::istream& stream) {
        auto allocator = svs::io::MemoryStreamAllocator<float>(stream);

        // Allocate 3 times, each time 2 floats
        auto* p0 = allocator.allocate(2);
        auto* p1 = allocator.allocate(2);
        auto* p2 = allocator.allocate(2);

        // Verify pointers are contiguous
        CATCH_REQUIRE(p1 == p0 + 2);
        CATCH_REQUIRE(p2 == p1 + 2);
        CATCH_REQUIRE(p2 == p0 + 4);

        // Verify data integrity
        CATCH_REQUIRE(p0[0] == data[0]);
        CATCH_REQUIRE(p0[1] == data[1]);
        CATCH_REQUIRE(p1[0] == data[2]);
        CATCH_REQUIRE(p1[1] == data[3]);
        CATCH_REQUIRE(p2[0] == data[4]);
        CATCH_REQUIRE(p2[1] == data[5]);
    };

    CATCH_SECTION("mmstream") {
        auto path = svs_test::prepare_temp_directory_v2() / "allocator_contiguous.bin";
        {
            auto out = std::ofstream(path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(data), sizeof(data));
        }
        auto stream = svs::io::mmstream(path);
        run_allocator_checks(stream);
    }

    CATCH_SECTION("ispanstream") {
        auto bytes = std::span<char>{reinterpret_cast<char*>(data), sizeof(data)};
        auto stream = svs::io::ispanstream(bytes);
        run_allocator_checks(stream);
    }

    CATCH_SECTION("std::stringstream") {
        auto stream = std::stringstream(std::ios::in | std::ios::out | std::ios::binary);
        stream.write(reinterpret_cast<const char*>(data), sizeof(data));
        run_allocator_checks(stream);
    }
}
