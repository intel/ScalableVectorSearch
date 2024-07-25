#include "svs-benchmark/datasets.h"
#include "svs-benchmark/benchmark.h"

#include "fmt/core.h"
#include "fmt/ostream.h"

#include <array>
#include <ostream>
#include <string_view>

namespace svsbenchmark {
namespace {

template <typename T, size_t N>
constexpr std::array<std::string_view, N> get_names(const std::array<T, N>& x) {
    auto result = std::array<std::string_view, N>{};
    for (size_t i = 0; i < N; ++i) {
        result[i] = name(x[i]);
    }
    return result;
}

class DatasetDoc : public Benchmark {
  public:
    DatasetDoc() = default;

    virtual std::string do_name() const override { return "dataset_format_documentation"; }
    virtual int do_run(std::span<const std::string_view> SVS_UNUSED(args)) const override {
        Dataset::describe_layout(std::cout);
        return 0;
    }
};

} // namespace

// Documentation registration.
void register_dataset_documentation(svsbenchmark::ExecutableDispatcher& exe) {
    exe.register_executable(std::make_unique<DatasetDoc>());
}

void Uncompressed::describe_layout(std::ostream& stream) {
    constexpr std::string_view PROTOTYPE = R"(
Uncompressed data with a proto-type layout of as shown below:

{}

The field `data_type` can be any one of the SVS defined data-types, though backends may
specialize on this field.)";

    auto table = svs::lib::save_to_table(Uncompressed::example());
    fmt::print(stream, PROTOTYPE, fmt::streamed(table));
}

// Full dataset documentation.
void Dataset::describe_layout(std::ostream& stream) {
    constexpr std::string_view PROTOTYPE = R"(
A multi-level TOML file where the first level describes the kind of dataset and the second
level is a layout to that specific dataset. An example is shown below.

{}

The value of "kind" can take on of the following values: {}.
The layout of the sub-table corresponding to each value is described below.)";
    auto example = svs::lib::save_to_table(Dataset::example());
    auto kinds = std::vector<std::string_view>();
    svs::lib::for_each_type(
        svsbenchmark::DatasetTypes(),
        [&]<typename T>(svs::lib::Type<T>) { kinds.push_back(T::name); }
    );

    // Main Documentation.
    fmt::print(stream, PROTOTYPE, fmt::streamed(example), fmt::join(kinds, ", "));

    // Sub Documentations.
    svs::lib::for_each_type(
        svsbenchmark::DatasetTypes(),
        [&]<typename T>(svs::lib::Type<T>) {
            auto str = fmt::format("# KIND: {} #", T::name);
            auto bars = std::string(str.size(), '#');
            fmt::print(stream, "\n\n{}\n{}\n{}\n", bars, str, bars);
            T::describe_layout(stream);
        }
    );
    fmt::print(stream, "\n");
}

} // namespace svsbenchmark
