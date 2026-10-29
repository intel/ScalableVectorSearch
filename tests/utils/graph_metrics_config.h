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

#pragma once

#include "fmt/core.h"
#include "toml++/toml.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace graph_metrics {

namespace fs = std::filesystem;

struct Options {
    size_t total = 0;
    size_t dimensions = 0;
    size_t threads = 0;
    size_t degree = 0;
    size_t window = 0;
    size_t max_candidates = 0;
    float alpha = 0;
    std::string graph_type;
    std::vector<std::string> build_methods;
    size_t batch_size = 0;
    bool recompute_entry_point = false;
    bool use_full_search_history = true;
};

struct Input {
    std::string name;
    fs::path path;
    std::string source_type;
    std::string distance;
    size_t source_vectors = 0;
    std::string kind;
    std::string generation_json;
    std::optional<uint32_t> seed;
    std::string distribution;
    double offset = 0;
    double scale = 1;
    std::string sha256;
};

struct Configuration {
    fs::path path;
    fs::path output_json;
    std::vector<std::pair<Options, Input>> inputs;
};

inline void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T> T setting(const toml::table& table, std::string_view key) {
    auto value = table[key].template value_exact<T>();
    require(value.has_value(), fmt::format("Missing or invalid prepared setting: {}", key));
    return *value;
}

inline size_t count_setting(
    const toml::table& table,
    std::string_view key,
    size_t maximum = std::numeric_limits<size_t>::max()
) {
    const auto value = setting<int64_t>(table, key);
    require(
        value > 0 && static_cast<uint64_t>(value) <= maximum,
        fmt::format("Prepared {} must be in [1, {}]", key, maximum)
    );
    return static_cast<size_t>(value);
}

// The Python launcher owns the public JSON schema and cache validation. This
// private TOML protocol reuses SVS's parser without adding a C++ JSON dependency.
inline Configuration read_prepared(const fs::path& filename) {
    const auto table = toml::parse_file(filename.string());
    const auto version = setting<int64_t>(table, "protocol_version");
    require(
        version == 1, fmt::format("Unsupported prepared protocol {}; expected 1", version)
    );
    Configuration config;
    config.path = setting<std::string>(table, "configuration_file");
    config.output_json = setting<std::string>(table, "output_json");
    Options common;
    common.threads = count_setting(table, "threads");
    common.graph_type = setting<std::string>(table, "graph_type");
    require(
        common.graph_type == "both" || common.graph_type == "concurrent" ||
            common.graph_type == "non_concurrent",
        "Prepared graph_type must be both, concurrent, or non_concurrent"
    );
    const auto* methods = table["build_methods"].as_array();
    require(
        methods && !methods->empty(), "Prepared build_methods must be a nonempty array"
    );
    std::set<std::string> seen;
    for (const auto& node : *methods) {
        const auto method = node.value_exact<std::string>();
        require(
            method && (*method == "sample_by_sample" || *method == "single_batch" ||
                       *method == "batches"),
            "Prepared build method must be sample_by_sample, single_batch, or batches"
        );
        require(seen.insert(*method).second, "Duplicate prepared build method: " + *method);
        common.build_methods.push_back(*method);
    }
    require(
        common.graph_type != "non_concurrent" ||
            seen != std::set<std::string>{"sample_by_sample"},
        "sample_by_sample requires a concurrent index"
    );
    common.batch_size = count_setting(table, "batch_size");
    common.recompute_entry_point = setting<bool>(table, "recompute_entry_point");
    common.degree =
        count_setting(table, "degree", std::numeric_limits<uint32_t>::max() - 1);
    common.window = count_setting(table, "window");
    common.max_candidates = count_setting(table, "max_candidates");
    require(
        common.max_candidates >= common.window && common.window >= common.degree,
        "Prepared parameters require max_candidates >= window >= degree"
    );
    common.use_full_search_history = setting<bool>(table, "use_full_search_history");
    const auto* datasets = table["datasets"].as_array();
    require(datasets && !datasets->empty(), "Prepared datasets must be a nonempty array");
    for (const auto& node : *datasets) {
        const auto* dataset = node.as_table();
        require(dataset != nullptr, "Prepared dataset must be a table");
        auto options = common;
        options.total =
            count_setting(*dataset, "vectors", std::numeric_limits<uint32_t>::max() - 1);
        options.dimensions =
            count_setting(*dataset, "dimensions", std::numeric_limits<uint32_t>::max());
        require(
            options.dimensions <=
                (std::numeric_limits<size_t>::max() / options.total - 4) / sizeof(float),
            "Prepared vector data size overflows size_t"
        );
        Input input;
        input.name = setting<std::string>(*dataset, "name");
        input.path = setting<std::string>(*dataset, "path");
        input.distance = setting<std::string>(*dataset, "distance");
        require(
            input.distance == "L2" || input.distance == "MIP",
            "Prepared distance must be L2 or MIP"
        );
        options.alpha = static_cast<float>(setting<double>(*dataset, "alpha"));
        require(
            std::isfinite(options.alpha) &&
                (input.distance == "L2" ? options.alpha >= 1
                                        : options.alpha > 0 && options.alpha <= 1),
            "Prepared alpha must be finite and >= 1 for L2 or in (0, 1] for MIP"
        );
        input.source_type = setting<std::string>(*dataset, "source_type");
        require(
            input.source_type == "float32" || input.source_type == "float16" ||
                input.source_type == "uint8" || input.source_type == "int8",
            "Prepared source_type must be float32, float16, uint8, or int8"
        );
        input.source_vectors = count_setting(*dataset, "source_vectors");
        require(
            input.source_vectors >= options.total,
            "Prepared dataset has fewer rows than requested"
        );
        input.kind = setting<std::string>(*dataset, "kind");
        require(
            input.kind == "file" || input.kind == "synthetic",
            "Prepared kind must be file or synthetic"
        );
        input.generation_json = setting<std::string>(*dataset, "generation_json");
        input.sha256 = setting<std::string>(*dataset, "sha256");
        if (input.kind == "synthetic") {
            const auto seed = setting<int64_t>(*dataset, "seed");
            require(
                seed >= 0 &&
                    static_cast<uint64_t>(seed) <= std::numeric_limits<uint32_t>::max(),
                "Prepared seed must be in [0, 4294967295]"
            );
            input.seed = static_cast<uint32_t>(seed);
            input.distribution = setting<std::string>(*dataset, "distribution");
            require(
                input.distribution == "normal" || input.distribution == "uniform",
                "Prepared distribution must be normal or uniform"
            );
            input.offset = setting<double>(*dataset, "offset");
            input.scale = setting<double>(*dataset, "scale");
            require(
                std::isfinite(input.offset) && std::isfinite(input.scale) &&
                    input.scale > 0,
                "Prepared synthetic offset must be finite and scale must be finite and > 0"
            );
        }
        config.inputs.emplace_back(std::move(options), std::move(input));
    }
    return config;
}

inline void redirect_stream(FILE* stream, const fs::path& path) {
    fs::create_directories(path.parent_path());
    const int fd = open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0666);
    require(fd >= 0, "Cannot open output: " + path.string());
    const bool success = std::fflush(stream) == 0 && dup2(fd, fileno(stream)) >= 0;
    if (fd != fileno(stream)) {
        close(fd);
    }
    require(success, "Cannot redirect output: " + path.string());
}

class TemporaryFile {
  public:
    fs::path path;
    explicit TemporaryFile(const fs::path& destination) {
        fs::create_directories(destination.parent_path());
        auto pattern = destination.string() + ".tmp.XXXXXX";
        const int fd = mkstemp(pattern.data());
        require(fd >= 0, "Cannot create temporary file for " + destination.string());
        close(fd);
        path = pattern;
    }
    TemporaryFile(const TemporaryFile&) = delete;
    TemporaryFile& operator=(const TemporaryFile&) = delete;
    ~TemporaryFile() {
        std::error_code ignored;
        if (!path.empty()) {
            fs::remove(path, ignored);
        }
    }
    void commit(const fs::path& destination) {
        fs::rename(path, destination);
        path.clear();
    }
};

inline void generate_vectors(const Options& options, const Input& input) {
    static_assert(std::endian::native == std::endian::little);
    require(
        input.kind == "synthetic" && input.source_type == "float32" &&
            input.seed.has_value(),
        "Generation requires a synthetic float32 dataset with a seed"
    );
    std::ofstream stream{input.path, std::ios::binary};
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    const auto dimensions = static_cast<uint32_t>(options.dimensions);
    std::vector<float> row(options.dimensions);
    std::mt19937 generator{*input.seed};
    const bool normal = input.distribution == "normal";
    const double offset = input.offset;
    const double scale = input.scale;
    bool has_spare = false;
    double spare = 0;
    for (size_t i = 0; i < options.total; ++i) {
        for (float& coordinate : row) {
            double sample;
            if (!normal) {
                sample = static_cast<double>(generator()) / 4294967296.0;
            } else if (has_spare) {
                sample = spare;
                has_spare = false;
            } else {
                // Explicit transform avoids std::normal_distribution's implementation-
                // dependent algorithm. Saved files remain authoritative across launches.
                const double u = (static_cast<double>(generator()) + 1) / 4294967297.0;
                const double v = static_cast<double>(generator()) / 4294967296.0;
                const double radius = std::sqrt(-2 * std::log(u));
                const double angle = 2 * std::numbers::pi_v<double> * v;
                sample = radius * std::cos(angle);
                spare = radius * std::sin(angle);
                has_spare = true;
            }
            const double value = offset + scale * sample;
            require(
                std::isfinite(value) &&
                    std::abs(value) <= std::numeric_limits<float>::max(),
                "Synthetic coordinate is outside finite float32 range"
            );
            coordinate = static_cast<float>(value);
        }
        stream.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
        stream.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }
    stream.close();
}

} // namespace graph_metrics
