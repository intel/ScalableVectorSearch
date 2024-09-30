/*
 * Copyright (C) 2024 Intel Corporation
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

#pragma once

#include "svs-benchmark/benchmark.h"

#include "svs/lib/datatype.h"
#include "svs/lib/dispatcher.h"
#include "svs/lib/saveload.h"
#include "svs/third-party/fmt.h"

#include <ostream>
#include <string_view>

namespace svsbenchmark {

template <typename T> struct Parser;

struct KeyEqual {
    // Use the `key_equal` to model dataset descriptions that are expected to form an
    // equivalent class in terms of recall.
    //
    // Inother words, changes that affect performance should not cause arguments to not
    // be `key_equal`.
    //
    // We use this function when looking up previously generated results.
    template <typename T> bool operator()(const T& x, const T& y) const {
        // find via ADL.
        return key_equal(x, y);
    }
};

inline constexpr KeyEqual by_key_equal{};

// An executable that displays documentation for dataset types.
void register_dataset_documentation(svsbenchmark::ExecutableDispatcher&);

/////
///// Uncompressed Descriptor
/////

/// Regular old uncompressed data.
struct Uncompressed {
  public:
    constexpr static std::string_view name = "uncompressed";

    /// The data type of the
    svs::DataType data_type_;

  public:
    explicit Uncompressed(svs::DataType data_type)
        : data_type_{data_type} {}

    // Equality
    friend bool operator==(Uncompressed, Uncompressed) = default;
    friend bool key_equal(Uncompressed x, Uncompressed y) { return x == y; }

    static Uncompressed example() { return Uncompressed(svs::DataType::float16); }

    // Saving and Loading
    constexpr static svs::lib::Version save_version{0, 0, 0};
    constexpr static std::string_view serialization_schema =
        "benchmark_dataset_uncompressed";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema, save_version, {SVS_LIST_SAVE_(data_type)}
        );
    }

    static Uncompressed load(const svs::lib::ContextFreeLoadTable& table) {
        return Uncompressed{SVS_LOAD_MEMBER_AT_(table, data_type)};
    }

    // Describe the layout of the serialized TOML file.
    static void describe_layout(std::ostream&);
};

/////
///// LVQ Descriptor
/////

enum class LVQPackingStrategy { Sequential, Turbo16x8, Turbo16x4 };
constexpr std::array<LVQPackingStrategy, 3> all_lvq_strategies{
    LVQPackingStrategy::Sequential,
    LVQPackingStrategy::Turbo16x8,
    LVQPackingStrategy::Turbo16x4};

constexpr std::string_view name(LVQPackingStrategy s) {
    switch (s) {
        using enum LVQPackingStrategy;
        case Sequential: {
            return "sequential";
        }
        case Turbo16x8: {
            return "turbo<16x8>";
        }
        case Turbo16x4: {
            return "turbo<16x4>";
        }
    }
    throw ANNEXCEPTION("Unreachable!");
}

template <> struct Parser<LVQPackingStrategy> {
    static constexpr LVQPackingStrategy parse(std::string_view str) {
        using enum LVQPackingStrategy;
        if (constexpr auto s = name(Sequential); s == str) {
            return Sequential;
        }
        if (constexpr auto s = name(Turbo16x8); s == str) {
            return Turbo16x8;
        }
        if (constexpr auto s = name(Turbo16x4); s == str) {
            return Turbo16x4;
        }
        throw ANNEXCEPTION("Cannot parse {} as a LVQPackingStrategy!", str);
    }
};

} // namespace svsbenchmark

/// Saving and Loading Overrides
template <> struct svs::lib::Saver<svsbenchmark::LVQPackingStrategy> {
    static svs::lib::SaveNode save(svsbenchmark::LVQPackingStrategy s) { return name(s); }
};

template <> struct svs::lib::Loader<svsbenchmark::LVQPackingStrategy> {
    using T = svsbenchmark::LVQPackingStrategy;
    using toml_type = toml::value<std::string>;
    static T load(const toml_type& v) { return svsbenchmark::Parser<T>::parse(v.get()); }
};

namespace svsbenchmark {

struct LVQ {
  public:
    size_t primary_;
    size_t residual_;
    LVQPackingStrategy strategy_;

  public:
    static constexpr std::string_view name = "lvq";

    LVQ(size_t primary, size_t residual, LVQPackingStrategy strategy)
        : primary_{primary}
        , residual_{residual}
        , strategy_{strategy} {}

    // example
    static LVQ example() { return LVQ(4, 8, LVQPackingStrategy::Sequential); }

    // equality
    friend bool operator==(LVQ x, LVQ y) = default;
    friend bool key_equal(LVQ x, LVQ y) {
        return x.primary_ == y.primary_ && x.residual_ == y.residual_;
    }

    // saving and loading
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "benchmark_dataset_lvq";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(primary), SVS_LIST_SAVE_(residual), SVS_LIST_SAVE_(strategy)}
        );
    }

    static LVQ load(const svs::lib::ContextFreeLoadTable& table) {
        return LVQ(
            SVS_LOAD_MEMBER_AT_(table, primary),
            SVS_LOAD_MEMBER_AT_(table, residual),
            SVS_LOAD_MEMBER_AT_(table, strategy)
        );
    }

    // Describe the layout of the serialized TOML file.
    static void describe_layout(std::ostream&);
};

enum class LeanVecKind { float16, float32, lvq4, lvq8 };
constexpr std::array<LeanVecKind, 4> all_leanvec_kinds{
    LeanVecKind::float16, LeanVecKind::float32, LeanVecKind::lvq4, LeanVecKind::lvq8};

constexpr std::string_view name(LeanVecKind s) {
    switch (s) {
        using enum LeanVecKind;
        case float16: {
            return "float16";
        }
        case float32: {
            return "float32";
        }
        case lvq4: {
            return "lvq4";
        }
        case lvq8: {
            return "lvq8";
        }
    }
    throw ANNEXCEPTION("Unreachable!");
}

template <> struct Parser<LeanVecKind> {
    static constexpr LeanVecKind parse(std::string_view str) {
        using enum LeanVecKind;
        if (constexpr auto s = name(float16); s == str) {
            return float16;
        }
        if (constexpr auto s = name(float32); s == str) {
            return float32;
        }
        if (constexpr auto s = name(lvq4); s == str) {
            return lvq4;
        }
        if (constexpr auto s = name(lvq8); s == str) {
            return lvq8;
        }
        throw ANNEXCEPTION("Cannot parse {} as a LeanVecKind!", str);
    }
};
} // namespace svsbenchmark

template <> struct svs::lib::Saver<svsbenchmark::LeanVecKind> {
    static svs::lib::SaveNode save(svsbenchmark::LeanVecKind x) { return name(x); }
};

template <> struct svs::lib::Loader<svsbenchmark::LeanVecKind> {
    using T = svsbenchmark::LeanVecKind;
    using toml_type = toml::value<std::string>;
    static T load(const toml_type& v) { return svsbenchmark::Parser<T>::parse(v.get()); }
};

namespace svsbenchmark {

struct LeanVec {
  public:
    LeanVecKind primary_;
    LeanVecKind secondary_;
    size_t leanvec_dims_;
    std::optional<std::filesystem::path> data_matrix_;
    std::optional<std::filesystem::path> query_matrix_;

  public:
    static constexpr std::string_view name = "leanvec";

    LeanVec(
        LeanVecKind primary,
        LeanVecKind secondary,
        size_t leanvec_dims,
        const std::optional<std::filesystem::path>& data_matrix = {},
        const std::optional<std::filesystem::path>& query_matrix = {}
    )
        : primary_{primary}
        , secondary_{secondary}
        , leanvec_dims_{leanvec_dims}
        , data_matrix_{data_matrix}
        , query_matrix_{query_matrix} {
        if (data_matrix_.has_value() != query_matrix_.has_value()) {
            throw ANNEXCEPTION("Either provide both the matrices or provide none of them!");
        }
    }

    // equality
    friend bool operator==(const LeanVec&, const LeanVec&) = default;
    friend bool key_equal(const LeanVec& x, const LeanVec& y) {
        return (x.primary_ == y.primary_) && (x.secondary_ == y.secondary_) &&
               (x.leanvec_dims_ == y.leanvec_dims_) &&
               // For matrices, only check the presence of value
               (x.data_matrix_.has_value() == y.data_matrix_.has_value()) &&
               (x.query_matrix_.has_value() == y.query_matrix_.has_value());
    }

    // example
    static LeanVec example() {
        return LeanVec(LeanVecKind::lvq8, LeanVecKind::float16, 192);
    }

    // Saving and Loading
    // Version History:
    // * v0.0.1 (Breaking): Added `data_matrix` and `query_matrix` filepath fields for
    //   optionally providing externally provided transformation matrices.
    //
    //   Empty paths denote no such external matrix is desired.
    static constexpr svs::lib::Version save_version{0, 0, 1};
    static constexpr std::string_view serialization_schema = "benchmark_dataset_leanvec";
    svs::lib::SaveTable save() const {
        auto data_matrix = data_matrix_.value_or("");
        auto query_matrix = query_matrix_.value_or("");
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(primary),
             SVS_LIST_SAVE_(secondary),
             SVS_LIST_SAVE_(leanvec_dims),
             {"data_matrix", svs::lib::save(data_matrix)},
             {"query_matrix", svs::lib::save(query_matrix)}}
        );
    }

    static LeanVec load(
        const svs::lib::ContextFreeLoadTable& table,
        const std::optional<std::filesystem::path>& root
    ) {
        // Processing pipeline for file-paths extracted from the TOML file.
        using optional_type = std::optional<std::filesystem::path>;
        auto process = [&root](std::filesystem::path path) -> optional_type {
            if (path.empty()) {
                return std::nullopt;
            }
            if (root && path.is_relative()) {
                return optional_type(*root / path);
            }
            return optional_type(std::move(path));
        };

        auto data_matrix_file =
            svs::lib::load_at<std::filesystem::path>(table, "data_matrix");
        auto data_matrix = process(std::move(data_matrix_file));
        auto query_matrix_file =
            svs::lib::load_at<std::filesystem::path>(table, "query_matrix");
        auto query_matrix = process(std::move(query_matrix_file));

        return LeanVec(
            SVS_LOAD_MEMBER_AT_(table, primary),
            SVS_LOAD_MEMBER_AT_(table, secondary),
            SVS_LOAD_MEMBER_AT_(table, leanvec_dims),
            std::move(data_matrix),
            std::move(query_matrix)
        );
    }

    // Describe the layout of the serialized TOML file.
    static void describe_layout(std::ostream&);
};

using DatasetTypes = svs::lib::Types<Uncompressed, LVQ, LeanVec>;
using DatasetVariant = std::variant<Uncompressed, LVQ, LeanVec>;

template <typename T>
concept ValidDatasetSource = svs::lib::in<T>(DatasetTypes{});

struct Dataset {
  public:
    std::variant<Uncompressed, LVQ, LeanVec> kinds_;

  public:
    template <ValidDatasetSource T>
    Dataset(T kind)
        : kinds_{kind} {}

    template <ValidDatasetSource T, typename By = KeyEqual>
    bool match(T x, const By& by = {}) const {
        return std::visit(
            [&](auto y) {
                if constexpr (std::is_same_v<decltype(y), T>) {
                    return by(x, y);
                } else {
                    return false;
                }
            },
            kinds_
        );
    }

    static Dataset example() { return Dataset(Uncompressed(svs::DataType::float16)); }

    // Saving and Loading.
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "benchmark_dataset_abstract";
    svs::lib::SaveTable save() const {
        auto node = std::visit([&](auto y) { return svs::lib::save(y); }, kinds_);
        std::string_view kind =
            std::visit<std::string_view>([&](auto y) { return decltype(y)::name; }, kinds_);

        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {{"kind", svs::lib::save(kind)}, {"dataset", std::move(node)}}
        );
    }

    static Dataset load(
        const svs::lib::ContextFreeLoadTable& table,
        const std::optional<std::filesystem::path> root
    ) {
        auto kind = svs::lib::load_at<std::string>(table, "kind");

        // TODO: It would be nicer for variants to have first-class support in the saving
        // and loading framework.
        auto loaded = std::optional<Dataset>();
        svs::lib::for_each_type(DatasetTypes(), [&]<typename T>(svs::lib::Type<T>) {
            if (T::name == kind) {
                // If we are loading a LeanVec dataset prototype, then forward the root path
                // to its loader.
                if constexpr (std::is_same_v<T, LeanVec>) {
                    loaded.emplace(svs::lib::load_at<T>(table, "dataset", root));
                } else {
                    loaded.emplace(svs::lib::load_at<T>(table, "dataset"));
                }
            }
        });
        if (!loaded) {
            throw ANNEXCEPTION("Unknown dataset kind {}!", kind);
        }
        return loaded.value();
    }

    // Describe the layout of the serialized TOML file.
    static void describe_layout(std::ostream&);
};

} // namespace svsbenchmark
