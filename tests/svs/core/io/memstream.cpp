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

    auto buf = svs::io::mmstreambuf(path);
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

    auto buf = svs::io::mmstreambuf(path);
    CATCH_REQUIRE(buf.is_open());
    CATCH_REQUIRE(buf.size() == 0);
    CATCH_REQUIRE(buf.sgetc() == std::char_traits<char>::eof());
}

CATCH_TEST_CASE("mmstreambuf supports move operations", "[core][io][mmap]") {
    auto path = write_file("mmstream_move.bin", "abcdef");

    auto source = svs::io::mmstreambuf(path);
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

    CATCH_REQUIRE(svs::io::current_ptr<char>(stream) == nullptr);

    stream.seekg(0, std::ios_base::beg);
    CATCH_REQUIRE(svs::io::current_ptr<char>(stream) == b);

    stream.seekg(3, std::ios_base::beg);
    CATCH_REQUIRE(svs::io::current_ptr<char>(stream) == b + 3);

    stream.close();
    CATCH_REQUIRE(svs::io::current_ptr<char>(stream) == nullptr);

    auto empty_path = create_empty_file("mmstream_ptrs_empty.bin");
    auto empty_stream = svs::io::mmstream(empty_path);
    CATCH_REQUIRE(svs::io::current_ptr<char>(empty_stream) == nullptr);
}

CATCH_TEST_CASE("mmstream open throws on missing file", "[core][io][mmap]") {
    auto missing = svs_test::prepare_temp_directory_v2() / "mmstream_missing.bin";

    auto buf = svs::io::mmstreambuf{};
    CATCH_REQUIRE_THROWS_AS(buf.open(missing), std::system_error);

    auto stream = svs::io::mmstream{};
    CATCH_REQUIRE_THROWS_AS(stream.open(missing), std::system_error);
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

    // ---- empty mmstream: in-memory but empty, must return nullptr ----
    {
        auto empty_path = create_empty_file("memstream_ptr_empty.bin");
        auto empty_mm = svs::io::mmstream(empty_path);
        CATCH_REQUIRE(svs::io::is_memory_stream(empty_mm));
        CATCH_REQUIRE(svs::io::current_ptr<char>(empty_mm) == nullptr);
    }

    // ---- empty std::istringstream: in-memory but empty, must return nullptr ----
    {
        auto empty_iss = std::istringstream("");
        CATCH_REQUIRE(svs::io::is_memory_stream(empty_iss));
        CATCH_REQUIRE(svs::io::current_ptr<char>(empty_iss) == nullptr);
    }
}
