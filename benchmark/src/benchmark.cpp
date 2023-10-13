// svs-benchmark
#include "svs-benchmark/benchmark.h"

// svs
#include "svs/third-party/toml.h"

// third-party
#include "fmt/ostream.h"
#include "fmt/std.h"

// stl
#include <filesystem>
#include <optional>

namespace svsbenchmark {

void atomic_save(const toml::table& table, const std::filesystem::path& path) {
    auto temp_filename = path.stem();
    temp_filename += "_temp";
    temp_filename += path.extension();
    auto temppath = path.parent_path() / temp_filename;
    // Write the contents of the table to a temporary file.
    // Once the write is complete, use an atomic move to replace the actual destination
    // file.
    {
        auto ostream = svs::lib::open_write(temppath);
        ostream << table << '\n';
    }
    std::filesystem::rename(temppath, path);
}

void append_or_create(toml::table& table, const toml::table& data, std::string_view key) {
    auto itr = table.find(key);
    if (itr != table.end()) {
        // Get the node at the key.
        // It *should* be an array. If it isn't, abort!
        // Using a reference here is valid because we've checked that the iterator is valid.
        auto& node = itr->second;
        if (auto* array = node.as_array(); array) {
            array->push_back(data);
        } else {
            throw ANNEXCEPTION("Expected TOML node at path {} to be an array!", key);
        }
    } else {
        // Create and insert a new array.
        table.insert(key, toml::array(data));
    }
}

std::filesystem::path extract_filename(
    const toml::table& table,
    std::string_view key,
    const std::optional<std::filesystem::path>& root
) {
    auto filename = svs::lib::load_at<std::filesystem::path>(table, key);
    bool hasroot = root.has_value();
    if (hasroot) {
        filename = *root / filename;
    }
    if (!std::filesystem::exists(filename)) {
        throw ANNEXCEPTION(
            "Could not find {}file {} (parsed from {})!",
            hasroot ? "qualified " : "",
            filename,
            fmt::streamed(table[key].node()->source())
        );
    }
    return filename;
}

} // namespace svsbenchmark
