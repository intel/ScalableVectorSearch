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

void LVQ::describe_layout(std::ostream& stream) {
    constexpr std::string_view PROTOTYPE = R"(
LVQ compressed data with a proto-type layout of as shown below:

{}

Where:
* `primary` is the number of bits for the primary dataset.
* `residual` is the number of bits in the residual (can be 0).
* `strategy` is the packing strategy to used. It can be one of {}.)";

    auto table = svs::lib::save_to_table(LVQ::example());
    fmt::print(
        stream,
        PROTOTYPE,
        fmt::streamed(table),
        fmt::join(get_names(svsbenchmark::all_lvq_strategies), ", ")
    );
}

void LeanVec::describe_layout(std::ostream& stream) {
    constexpr std::string_view PROTOTYPE = R"(
LeanVec dimensionality reducing data with a layout of as shown below:

{}

Where:
* `primary` is the kind of the primary dataset [see note 1 for valid values].
* `secondary` is the kind of the secondary dataset [see note 1 for valid values].
* `leanvec_dims` is the number of dimensions in the reduced primary dataset.
* `data_matrix` is the matrix used for data transformation [see note 2].
* `query_matrix` is the matrix used for query transformation [see note 2].
* Provide both the matrices or none. Providing one of them is not allowed.

Note 1: Argument `kind` is a string and can take one of the following values: {}
Note 2: These arguments are optional in the sense that providing an empty string defaults
    to internally generated PCA-based transformation matrices.

    It is required that either *both* matrices are supplied or *neither* are supplied.

    The paths for `data_matrix` and `query_matrix` can alias if the same transformation
    for queries and data is desired.
)";

    auto table = svs::lib::save_to_table(LeanVec::example());
    fmt::print(
        stream,
        PROTOTYPE,
        fmt::streamed(table),
        fmt::join(get_names(svsbenchmark::all_leanvec_kinds), ", ")
    );
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
