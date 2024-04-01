//! [example-all]

#include "svs/lib/dispatcher.h"
#include "svs/third-party/fmt.h"

#include "svsmain.h"

#include <string_view>

namespace {

std::optional<bool> parse_bool_nothrow(std::string_view arg) {
    if (arg == "true") {
        return true;
    }
    if (arg == "false") {
        return false;
    }
    return std::nullopt;
}

/// Parse a string as a boolean.
///
/// Throws a ``svs::ANNException`` if parsing fails.
bool parse_bool(std::string_view arg) {
    auto v = parse_bool_nothrow(arg);
    if (!v) {
        throw ANNEXCEPTION(
            "Cannot parse \"{}\" as a boolean value! Expected either \"true\" or "
            "\"false\".",
            arg
        );
    }
    return v.value();
}

/// Parse a string as a valid ``svs::DataType``.
svs::DataType parse_datatype(std::string_view arg) {
    auto type = svs::parse_datatype(arg);
    if (type == svs::DataType::undef) {
        throw ANNEXCEPTION("Cannot parse {} as an SVS datatype!", type);
    }
    return type;
}

/// Parse a string as an extent.
svs::lib::ExtentArg parse_extent_arg(const std::string& extent, bool enforce) {
    if (extent == "dynamic") {
        return svs::lib::ExtentArg{svs::Dynamic, enforce};
    }
    // Try to parse as an integer.
    return svs::lib::ExtentArg{std::stoull(extent), enforce};
}

std::string format_extent(size_t n) {
    return n == svs::Dynamic ? std::string("dynamic") : fmt::format("{}", n);
}

//! [specialization-1]
// A specialized method.
// SVS defines the dispatch conversion from `svs::DataType` to `svs::lib::Type`.
// This overload takes an additional `std::string` argument type.
template <typename A, typename B, size_t N>
void specialized(
    svs::lib::Type<A> a_type,
    svs::lib::Type<B> b_type,
    svs::lib::ExtentTag<N> SVS_UNUSED(extent_tag),
    const std::string& arg
) {
    // Convert `svs::lib::Type` to `svs::DataType`.
    svs::DataType a = a_type;
    svs::DataType b = b_type;
    fmt::print(
        "Specialized with string: {}, {}, {} with arg \"{}\"\n", a, b, format_extent(N), arg
    );
}
//! [specialization-1]

//! [specialization-2]
template <typename A, typename B, size_t N>
void specialized_alternative(
    svs::lib::Type<A> a_type,
    svs::lib::Type<B> b_type,
    svs::lib::ExtentTag<N> SVS_UNUSED(extent_tag),
    bool flag
) {
    // Convert `svs::lib::Type` to `svs::DataType`.
    svs::DataType a = a_type;
    svs::DataType b = b_type;
    fmt::print(
        "Specialized with flag: {}, {}, {} with arg \"{}\"\n", a, b, format_extent(N), flag
    );
}
//! [specialization-2]

//! [generic-fallback]
struct Converted {
    std::string value_;
};
void generic(
    svs::DataType a_type,
    svs::DataType b_type,
    svs::lib::ExtentTag<svs::Dynamic> SVS_UNUSED(extent_tag),
    Converted converted
) {
    fmt::print(
        "Generic: {}, {}, {} with arg \"{}\"\n",
        a_type,
        b_type,
        format_extent(svs::Dynamic),
        converted.value_
    );
}
//! [generic-fallback]

}

//! [converted-dispatch-conversion-rules]
// Define full-specializations for converting `std::string` and `bool` to `Converted`.
template <> struct svs::lib::DispatchConverter<std::string, Converted> {
    // Return ``true`` if this is a match. Otherwise, return false.
    //
    // To provide finer-grained control, an ``int64_t`` can be returned instead, where
    // negative values indicate an invalid match (a method that has an invalid match for
    // any argument must be discarded), and positive values indicate degree of matches
    // with lower numbers having higher priority.
    static bool match(const std::string& SVS_UNUSED(arg)) { return true; }

    // This argument is called when a method has been selected and we're ready to convert
    // the source argument to the destination argument and invoke the registered target.
    static Converted convert(const std::string& arg) { return Converted{arg}; }

    // Provide documentation regarding the values accepted by this conversion.
    static std::string_view description() { return "all-string-values"; }
};

template <> struct svs::lib::DispatchConverter<bool, Converted> {
    static bool match(bool SVS_UNUSED(arg)) { return true; }
    static Converted convert(bool arg) { return Converted{fmt::format("boolean {}", arg)}; }
    static std::string_view description() { return "all-boolean-values"; }
};
//! [converted-dispatch-conversion-rules]

namespace {

//! [dispatcher-definition]
// A variant of types given for the last argument of the methods we wish to dispatch to.
using Variant = std::variant<bool, std::string>;

// A dispatcher wrapping and dispatching to functions that return void and whose arguments
// are constructible using dispatcher conversion from the remaining types.
using Dispatcher =
    svs::lib::Dispatcher<void, svs::DataType, svs::DataType, svs::lib::ExtentArg, Variant>;
//! [dispatcher-definition]

//! [register-methods]
Dispatcher build_and_register() {
    // Default construction for a dispatcher.
    auto dispatcher = Dispatcher{};

    // When registering methods, argument documentation can be requested using this tag.
    // Note dispatch rules are not required to implement documentation, in which case
    // default documentation will be provided.
    constexpr auto build_docs = svs::lib::dispatcher_build_docs;

    // Register the desired specializations.
    dispatcher.register_target(build_docs, &specialized<float, float, svs::Dynamic>);
    dispatcher
        .register_target(build_docs, &specialized_alternative<float, float, svs::Dynamic>);
    dispatcher.register_target(build_docs, &specialized<uint32_t, uint8_t, 128>);

    // Register the dynamic fallback.
    dispatcher.register_target(build_docs, &generic);
    return dispatcher;
}

const Dispatcher& get_dispatcher() {
    // Only allocated and populate the dispatcher once.
    static Dispatcher dispatcher = build_and_register();
    return dispatcher;
}
//! [register-methods]

} // namespace

//! [print-help]
void print_help() {
    constexpr std::string_view help_template = R"(
Usage:
    (1) dispatcher type_a type_b dims enforce_dims arg
    (2) dispatcher --help

1. Run the dispatcher example.
   * type_a and type_b: must be parseable as s `svs::DataType`.
   * dims: The number of dimensions to dispatch on. Can either be an integer or the string
     "dynamic"
   * enforce_dims: Whether or not relaxation to dynamic dimensionality is allowed. Must
     either be "true" or "false"
   * arg: An additional string argument. If arg is either "true" or "false", it will be
     parsed as a boolean. Otherwise, it will remain as a string and be forwarded to the
     appropriate overload.

2. Print this help message.

Registered Specializations
--------------------------
{{ type A, type B, Extent, Last Argument }}

{}
)";
    const auto& dispatcher = get_dispatcher();
    auto method_docs = std::vector<std::string>();
    constexpr size_t nargs = Dispatcher::num_args();
    for (size_t i = 0, imax = dispatcher.size(); i < imax; ++i) {
        auto arg_docs = std::vector<std::string>();
        for (size_t j = 0; j < nargs; ++j) {
            arg_docs.push_back(dispatcher.description(i, j));
        }
        method_docs.push_back(fmt::format("{{ {} }}", fmt::join(arg_docs, ", ")));
    }

    fmt::print(help_template, fmt::join(method_docs, "\n"));
}
//! [print-help]

//! [main]
int svs_main(const std::vector<std::string>& args) {
    // Perform some very basic
    const size_t nargs = args.size();
    bool requested_help = std::any_of(args.begin(), args.end(), [](const auto& arg) {
        return arg == "--help" || arg == "help";
    });
    if (nargs != 6 || requested_help) {
        print_help();
        return 0;
    }

    // Parse the argument types.
    svs::DataType type_a = parse_datatype(args.at(1));
    svs::DataType type_b = parse_datatype(args.at(2));
    svs::lib::ExtentArg extent_arg = parse_extent_arg(args.at(3), parse_bool(args.at(4)));

    // Construct a variant according to the value of `arg`.
    const auto& arg = args.at(5);
    auto maybe_bool = parse_bool_nothrow(arg);
    auto variant = [&]() -> Variant {
        if (maybe_bool) {
            return maybe_bool.value();
        }
        return arg;
    }();

    // Instantiate the dispatcher and dispatch to the best fitting method.
    get_dispatcher().invoke(type_a, type_b, extent_arg, variant);
    return 0;
}
//! [main]

// Main helper.
SVS_DEFINE_MAIN();

//! [example-all]
