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
#include "nlohmann/json.hpp"
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif

namespace graph_metrics {

using Json = nlohmann::json;
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
    Json generation = nullptr;
    std::string sha256;
};

struct Configuration {
    fs::path path;
    fs::path output_json;
    fs::path output_log;
    std::vector<std::pair<Options, Input>> inputs;
};

inline void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline Json read_json(const fs::path& path) {
    std::ifstream stream{path};
    require(bool(stream), fmt::format("Cannot read JSON: {}", path.string()));
    // The parser normally keeps the last value of a duplicate key. Reject
    // duplicates so a misspecified configuration cannot silently change a run.
    std::vector<std::set<std::string>> keys;
    auto callback = [&](int, Json::parse_event_t event, Json& value) {
        if (event == Json::parse_event_t::object_start) {
            keys.emplace_back();
        } else if (event == Json::parse_event_t::key) {
            const auto key = value.get<std::string>();
            require(keys.back().insert(key).second, "Duplicate JSON key: " + key);
        } else if (event == Json::parse_event_t::object_end) {
            keys.pop_back();
        }
        return true;
    };
    return Json::parse(stream, callback);
}

inline void object_keys(
    const Json& value,
    std::string_view context,
    std::initializer_list<std::string_view> required,
    std::initializer_list<std::string_view> optional = {}
) {
    require(value.is_object(), fmt::format("{} must be an object", context));
    for (auto key : required) {
        require(value.contains(key), fmt::format("Missing {}.{}", context, key));
    }
    for (auto it = value.begin(); it != value.end(); ++it) {
        require(
            std::find(required.begin(), required.end(), it.key()) != required.end() ||
                std::find(optional.begin(), optional.end(), it.key()) != optional.end(),
            fmt::format("Unknown field {}.{}", context, it.key())
        );
    }
}

inline std::string text(const Json& value, std::string_view context) {
    require(value.is_string(), fmt::format("{} must be a string", context));
    auto result = value.get<std::string>();
    require(
        !result.empty() && result.find('\0') == std::string::npos,
        fmt::format("{} must be nonempty and contain no NUL", context)
    );
    return result;
}

inline size_t integer(
    const Json& value,
    std::string_view context,
    size_t minimum = 1,
    size_t maximum = std::numeric_limits<size_t>::max()
) {
    require(
        value.is_number_unsigned() && value.get<uint64_t>() >= minimum &&
            value.get<uint64_t>() <= maximum,
        fmt::format("{} must be an integer in [{}, {}]", context, minimum, maximum)
    );
    return value.get<size_t>();
}

inline double number(const Json& value, std::string_view context) {
    require(value.is_number(), fmt::format("{} must be a number", context));
    const auto result = value.get<double>();
    require(std::isfinite(result), fmt::format("{} must be finite", context));
    return result;
}

inline bool boolean(const Json& value, std::string_view context) {
    require(value.is_boolean(), fmt::format("{} must be a boolean", context));
    return value.get<bool>();
}

inline size_t available_threads() {
#ifdef __linux__
    cpu_set_t affinity;
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0 &&
        CPU_COUNT(&affinity) > 0) {
        return static_cast<size_t>(CPU_COUNT(&affinity));
    }
#endif
    return std::max(1u, std::thread::hardware_concurrency());
}

inline size_t element_size(const std::string& type) {
    if (type == "uint8" || type == "int8") {
        return 1;
    }
    if (type == "float16") {
        return 2;
    }
    require(type == "float32", "Unsupported source type: " + type);
    return 4;
}

inline fs::path resolve(const fs::path& directory, const Json& value) {
    return fs::weakly_canonical(directory / text(value, "path"));
}

inline fs::path metadata_path(const Input& input) {
    return input.path.string() + ".meta.json";
}

inline Configuration read_configuration(const fs::path& filename) {
    Configuration config;
    config.path = fs::canonical(filename);
    const auto directory = config.path.parent_path();
    const auto json = read_json(config.path);
    object_keys(
        json,
        "config",
        {"threads",
         "graph_type",
         "build_methods",
         "batch_size",
         "recompute_entry_point",
         "build_parameters",
         "datasets",
         "output_json"},
        {"output_log"}
    );
    Options common;
    const auto& threads = json.at("threads");
    if (threads.is_string()) {
        require(threads == "auto", "threads must be a positive integer or \"auto\"");
        common.threads = available_threads();
    } else {
        common.threads = integer(threads, "threads");
    }
    common.graph_type = text(json.at("graph_type"), "graph_type");
    require(
        common.graph_type == "concurrent" || common.graph_type == "non_concurrent" ||
            common.graph_type == "both",
        "graph_type must be concurrent, non_concurrent, or both"
    );
    const auto& methods = json.at("build_methods");
    require(
        methods.is_array() && !methods.empty(), "build_methods must be a nonempty array"
    );
    std::set<std::string> seen_methods;
    for (const auto& method : methods) {
        const auto name = text(method, "build_methods entry");
        require(
            name == "sample_by_sample" || name == "single_batch" || name == "batches",
            "Unknown build method: " + name
        );
        require(seen_methods.insert(name).second, "Duplicate build method: " + name);
        common.build_methods.push_back(name);
    }
    require(
        common.graph_type != "non_concurrent" || seen_methods.size() != 1 ||
            !seen_methods.contains("sample_by_sample"),
        "sample_by_sample requires a concurrent index"
    );
    common.batch_size = integer(json.at("batch_size"), "batch_size");
    common.recompute_entry_point =
        boolean(json.at("recompute_entry_point"), "recompute_entry_point");
    const auto& build = json.at("build_parameters");
    object_keys(
        build,
        "build_parameters",
        {"degree", "window", "max_candidates", "alpha"},
        {"use_full_search_history"}
    );
    common.degree =
        integer(build.at("degree"), "degree", 1, std::numeric_limits<uint32_t>::max() - 1);
    common.window = integer(build.at("window"), "window");
    common.max_candidates = integer(build.at("max_candidates"), "max_candidates");
    require(
        common.max_candidates >= common.window && common.window >= common.degree,
        "Require max_candidates >= window >= degree"
    );
    if (build.contains("use_full_search_history")) {
        common.use_full_search_history =
            boolean(build.at("use_full_search_history"), "use_full_search_history");
    }
    const auto& alpha = build.at("alpha");
    object_keys(alpha, "alpha", {"L2", "MIP"});
    const auto alpha_l2 = number(alpha.at("L2"), "alpha.L2");
    const auto alpha_mip = number(alpha.at("MIP"), "alpha.MIP");
    require(
        alpha_l2 >= 1 && alpha_l2 <= std::numeric_limits<float>::max() && alpha_mip > 0 &&
            alpha_mip <= 1 && static_cast<float>(alpha_mip) > 0,
        "Require alpha.L2 >= 1 and alpha.MIP in (0, 1], representable as float32"
    );
    config.output_json = resolve(directory, json.at("output_json"));
    if (json.contains("output_log")) {
        config.output_log = resolve(directory, json.at("output_log"));
    }
    const auto& datasets = json.at("datasets");
    require(datasets.is_array() && !datasets.empty(), "datasets must be a nonempty array");
    std::set<std::string> names;
    std::map<fs::path, Json> generators;
    std::set<fs::path> protected_paths{config.path};
    for (const auto& dataset : datasets) {
        object_keys(
            dataset, "dataset", {"name", "vectors", "dimensions", "distance", "source"}
        );
        auto options = common;
        Input input;
        input.name = text(dataset.at("name"), "dataset.name");
        require(names.insert(input.name).second, "Duplicate dataset name: " + input.name);
        options.total = integer(
            dataset.at("vectors"), "vectors", 1, std::numeric_limits<uint32_t>::max() - 1
        );
        options.dimensions = integer(
            dataset.at("dimensions"), "dimensions", 1, std::numeric_limits<uint32_t>::max()
        );
        require(
            options.dimensions <=
                (std::numeric_limits<size_t>::max() / options.total - 4) / sizeof(float),
            "Vector data size overflows size_t"
        );
        input.distance = text(dataset.at("distance"), "distance");
        require(
            input.distance == "L2" || input.distance == "MIP", "distance must be L2 or MIP"
        );
        options.alpha = static_cast<float>(input.distance == "L2" ? alpha_l2 : alpha_mip);
        const auto& source = dataset.at("source");
        require(source.is_object() && source.contains("type"), "Missing source.type");
        input.kind = text(source.at("type"), "source.type");
        if (input.kind == "file") {
            object_keys(source, "source", {"type", "path", "data_type"});
            input.source_type = text(source.at("data_type"), "source.data_type");
            element_size(input.source_type);
        } else {
            require(input.kind == "synthetic", "source.type must be file or synthetic");
            require(source.contains("distribution"), "Missing source.distribution");
            const auto distribution = text(source.at("distribution"), "distribution");
            Json parameters;
            if (distribution == "normal") {
                object_keys(
                    source,
                    "source",
                    {"type", "path", "distribution", "mean", "stddev", "seed"}
                );
                parameters = {
                    {"mean", number(source.at("mean"), "mean")},
                    {"stddev", number(source.at("stddev"), "stddev")}};
                require(parameters.at("stddev").get<double>() > 0, "stddev must be > 0");
            } else {
                require(
                    distribution == "uniform", "distribution must be normal or uniform"
                );
                object_keys(
                    source,
                    "source",
                    {"type", "path", "distribution", "lower", "upper", "seed"}
                );
                parameters = {
                    {"lower", number(source.at("lower"), "lower")},
                    {"upper", number(source.at("upper"), "upper")}};
                require(
                    parameters.at("lower").get<double>() <
                        parameters.at("upper").get<double>(),
                    "Require lower < upper"
                );
            }
            for (const auto& parameter : parameters) {
                require(
                    std::abs(parameter.get<double>()) <= std::numeric_limits<float>::max(),
                    "Synthetic parameters must fit float32"
                );
            }
            const auto seed =
                integer(source.at("seed"), "seed", 0, std::numeric_limits<uint32_t>::max());
            input.source_type = "float32";
            input.generation = {
                {"schema_version", 1},
                {"generator", "svs-mt19937-box-muller-v1"},
                {"vectors", options.total},
                {"dimensions", options.dimensions},
                {"data_type", input.source_type},
                {"distribution", distribution},
                {"parameters", parameters},
                {"seed", seed}};
        }
        input.path = resolve(directory, source.at("path"));
        protected_paths.insert(input.path);
        if (input.kind == "synthetic") {
            const auto [it, inserted] = generators.emplace(input.path, input.generation);
            require(
                inserted || it->second == input.generation,
                "Conflicting synthetic configurations for " + input.path.string()
            );
        }
        config.inputs.emplace_back(std::move(options), std::move(input));
    }
    for (const auto& [options, input] : config.inputs) {
        (void)options;
        if (input.kind == "synthetic") {
            for (const fs::path& sidecar :
                 {metadata_path(input), fs::path{input.path.string() + ".lock"}}) {
                require(
                    !protected_paths.contains(sidecar),
                    "Synthetic sidecar collides with a dataset or configuration: " +
                        sidecar.string()
                );
            }
        }
    }
    for (const auto& [options, input] : config.inputs) {
        (void)options;
        if (input.kind == "synthetic") {
            protected_paths.insert(metadata_path(input));
            protected_paths.insert(input.path.string() + ".lock");
        }
    }
    for (const auto& output : {config.output_json, config.output_log}) {
        require(
            output.empty() || !protected_paths.contains(output),
            "Output path collides with an input: " + output.string()
        );
        if (!output.empty() && fs::exists(output)) {
            for (const auto& input : protected_paths) {
                require(
                    !fs::exists(input) || !fs::equivalent(input, output),
                    "Output path aliases an input: " + output.string()
                );
            }
        }
    }
    require(config.output_json != config.output_log, "JSON and log paths must differ");
    require(
        config.output_log.empty() || !fs::exists(config.output_json) ||
            !fs::exists(config.output_log) ||
            !fs::equivalent(config.output_json, config.output_log),
        "JSON and log paths must not alias the same file"
    );
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

class Sha256 {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context_{
        EVP_MD_CTX_new(), EVP_MD_CTX_free};

  public:
    Sha256() {
        require(
            context_ && EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) == 1,
            "Cannot initialize SHA-256"
        );
    }
    void update(const void* data, size_t size) {
        require(EVP_DigestUpdate(context_.get(), data, size) == 1, "SHA-256 update failed");
    }
    std::string finish() {
        std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
        unsigned int size = 0;
        require(
            EVP_DigestFinal_ex(context_.get(), digest.data(), &size) == 1,
            "SHA-256 finalization failed"
        );
        std::string result;
        for (unsigned int i = 0; i < size; ++i) {
            result += fmt::format("{:02x}", digest[i]);
        }
        return result;
    }
};

inline std::string hash_prefix(const fs::path& path, size_t bytes) {
    std::ifstream stream{path, std::ios::binary};
    Sha256 hash;
    std::vector<char> buffer(1024 * 1024);
    while (bytes != 0) {
        const auto count = std::min(bytes, buffer.size());
        require(
            bool(stream.read(buffer.data(), count)),
            "Cannot read dataset for checksum: " + path.string()
        );
        hash.update(buffer.data(), count);
        bytes -= count;
    }
    return hash.finish();
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

class DatasetLock {
    int fd_;

  public:
    explicit DatasetLock(const fs::path& path)
        : fd_{open(path.c_str(), O_CREAT | O_RDWR, 0600)} {
        require(fd_ >= 0, "Cannot open dataset lock: " + path.string());
        int result;
        do {
            result = flock(fd_, LOCK_EX);
        } while (result != 0 && errno == EINTR);
        if (result != 0) {
            close(fd_);
            throw std::runtime_error("Cannot lock dataset: " + path.string());
        }
    }
    DatasetLock(const DatasetLock&) = delete;
    DatasetLock& operator=(const DatasetLock&) = delete;
    ~DatasetLock() { close(fd_); }
};

inline void prepare_synthetic(const Options& options, Input& input) {
    static_assert(std::endian::native == std::endian::little);
    fs::create_directories(input.path.parent_path());
    DatasetLock lock{input.path.string() + ".lock"};
    const auto metadata = metadata_path(input);
    const auto bytes = options.total * (4 + sizeof(float) * options.dimensions);
    if (fs::exists(input.path) || fs::exists(metadata)) {
        try {
            require(
                fs::is_regular_file(input.path) && fs::is_regular_file(metadata),
                "Incomplete synthetic dataset: both data and metadata must exist"
            );
            auto saved = read_json(metadata);
            require(
                saved.is_object() && saved.contains("sha256"), "Missing synthetic SHA-256"
            );
            const auto expected_hash = text(saved.at("sha256"), "synthetic sha256");
            saved.erase("sha256");
            require(saved == input.generation, "Synthetic metadata mismatch");
            require(fs::file_size(input.path) == bytes, "Synthetic dataset size mismatch");
            input.sha256 = hash_prefix(input.path, bytes);
            require(input.sha256 == expected_hash, "Synthetic dataset checksum mismatch");
            fmt::print(stderr, "Reusing synthetic dataset: {}\n", input.path.string());
            return;
        } catch (const std::exception& error) {
            // An obsolete or incomplete cache is recoverable. Keep the old files
            // until the replacement data and metadata have both been generated.
            fmt::print(
                stderr,
                "Regenerating synthetic dataset: {} ({})\n",
                input.path.string(),
                error.what()
            );
        }
    } else {
        fmt::print(stderr, "Generating synthetic dataset: {}\n", input.path.string());
    }
    TemporaryFile vectors{input.path};
    TemporaryFile sidecar{metadata};
    std::ofstream stream{vectors.path, std::ios::binary};
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    const auto dimensions = static_cast<uint32_t>(options.dimensions);
    std::vector<float> row(options.dimensions);
    std::mt19937 generator{input.generation.at("seed").get<uint32_t>()};
    const auto& parameters = input.generation.at("parameters");
    const bool normal = input.generation.at("distribution") == "normal";
    const double offset = parameters.at(normal ? "mean" : "lower").get<double>();
    const double scale = normal ? parameters.at("stddev").get<double>()
                                : parameters.at("upper").get<double>() - offset;
    bool has_spare = false;
    double spare = 0;
    Sha256 hash;
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
        hash.update(&dimensions, sizeof(dimensions));
        hash.update(row.data(), row.size() * sizeof(float));
    }
    stream.close();
    input.sha256 = hash.finish();
    auto saved = input.generation;
    saved["sha256"] = input.sha256;
    std::ofstream metadata_stream{sidecar.path};
    metadata_stream.exceptions(std::ios::failbit | std::ios::badbit);
    metadata_stream << saved.dump(2) << '\n';
    metadata_stream.close();
    vectors.commit(input.path);
    sidecar.commit(metadata);
}

inline void inspect_file(const Options& options, Input& input) {
    static_assert(std::endian::native == std::endian::little);
    const size_t row_bytes = 4 + element_size(input.source_type) * options.dimensions;
    std::ifstream stream{input.path, std::ios::binary};
    uint32_t dimensions = 0;
    require(
        bool(stream.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions))),
        "Cannot read dataset header: " + input.path.string()
    );
    const auto bytes = fs::file_size(input.path);
    require(
        dimensions == options.dimensions && bytes % row_bytes == 0,
        "Invalid vecs file dimensions or length: " + input.path.string()
    );
    input.source_vectors = bytes / row_bytes;
    require(input.source_vectors >= options.total, "Dataset has fewer rows than requested");
    if (input.sha256.empty()) {
        input.sha256 = hash_prefix(input.path, options.total * row_bytes);
    }
}

inline void prepare_inputs(Configuration& config) {
    // Preflight file inputs before spending time generating synthetic datasets.
    for (auto& [options, input] : config.inputs) {
        if (input.kind == "file") {
            inspect_file(options, input);
        }
    }
    for (auto& [options, input] : config.inputs) {
        if (input.kind == "synthetic") {
            prepare_synthetic(options, input);
            inspect_file(options, input);
        }
    }
}

} // namespace graph_metrics
