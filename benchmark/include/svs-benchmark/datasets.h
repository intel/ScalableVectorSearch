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

using DatasetTypes = svs::lib::Types<Uncompressed>;
using DatasetVariant = std::variant<Uncompressed>;

template <typename T>
concept ValidDatasetSource = svs::lib::in<T>(DatasetTypes{});

struct Dataset {
  public:
    std::variant<Uncompressed> kinds_;

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
                loaded.emplace(svs::lib::load_at<T>(table, "dataset"));
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
