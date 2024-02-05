/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

// Header under test
#include "svs/lib/dispatcher.h"

// catch2
#include "catch2/catch_test_macros.hpp"

/////
///// Dispatcher V2
/////

namespace lib = svs::lib;

namespace {

struct Uncopyable {
  public:
    Uncopyable(int64_t value)
        : value_{value} {}

    bool is_moved_from() const { return moved_from_; }

    // Special member functions.
    Uncopyable(const Uncopyable&) = delete;
    Uncopyable& operator=(const Uncopyable&) = delete;
    Uncopyable(Uncopyable&& other)
        : value_{other.value_} {
        other.moved_from_ = true;
    }

    Uncopyable& operator=(Uncopyable&& other) {
        value_ = other.value_;
        moved_from_ = false;
        other.moved_from_ = true;
        return *this;
    }

    ~Uncopyable() = default;

  public:
    int64_t value_;
    bool moved_from_ = false;
};

template <typename Left, typename Right, typename F>
void do_conversion_call(Left&& x, F&& f) {
    f(lib::dispatch_convert<Left, Right>(SVS_FWD(x)));
}

template <typename Left, typename Right, typename F>
void check_conversion_call(Left x, F&& f) {
    bool called = false;
    do_conversion_call<Left, Right>(SVS_FWD(x), [&f, &called](Right y) {
        called = true;
        f(SVS_FWD(y));
    });
    CATCH_REQUIRE(called);
}

#define TEST_ALL_COMBINATIONS(From, To, f)                                                 \
    CATCH_SECTION("Value to Value") { f.template operator()<From, To>(); }                 \
    CATCH_SECTION("Value to Const Ref") { f.template operator()<From, const To&>(); }      \
    CATCH_SECTION("Value to RValueRef") { f.template operator()<From, To&&>(); }           \
    CATCH_SECTION("Ref to ConstRef") { f.template operator()<From&, const To&>(); }        \
    CATCH_SECTION("Ref to Ref") { f.template operator()<From&, To&>(); }                   \
    CATCH_SECTION("ConstRef to ConstRef") {                                                \
        f.template operator()<const From&, const To&>();                                   \
    }                                                                                      \
    CATCH_SECTION("RValueRef to Value") { f.template operator()<From&&, To>(); }           \
    CATCH_SECTION("RValueRef to ConstRef") { f.template operator()<From&&, const To&>(); } \
    CATCH_SECTION("RValueRef to RValueRef") { f.template operator()<From&&, To&&>(); }

template <typename T>
inline constexpr bool is_mutable_reference_v =
    std::is_lvalue_reference_v<T> && !std::is_const_v<std::remove_reference_t<T>>;

template <typename F> using Signature = lib::detail::Signature<F>;

} // namespace

CATCH_TEST_CASE("Dispatcher2", "[lib][dispatcher2]") {
    CATCH_SECTION("Implicitly Dispatch Convertible") {
        CATCH_STATIC_REQUIRE(lib::ImplicitlyDispatchConvertible<size_t, size_t>);
        CATCH_STATIC_REQUIRE(lib::ImplicitlyDispatchConvertible<size_t, const size_t&>);
        CATCH_STATIC_REQUIRE(!lib::ImplicitlyDispatchConvertible<size_t, size_t&>);
        CATCH_STATIC_REQUIRE(lib::ImplicitlyDispatchConvertible<size_t, size_t&&>);

        CATCH_STATIC_REQUIRE(!lib::ImplicitlyDispatchConvertible<size_t&, size_t>);
        CATCH_STATIC_REQUIRE(lib::ImplicitlyDispatchConvertible<size_t&, const size_t&>);
        CATCH_STATIC_REQUIRE(lib::ImplicitlyDispatchConvertible<size_t&, size_t&>);
        CATCH_STATIC_REQUIRE(!lib::ImplicitlyDispatchConvertible<size_t&, size_t&&>);

        CATCH_STATIC_REQUIRE(!lib::ImplicitlyDispatchConvertible<const size_t&, size_t>);
        CATCH_STATIC_REQUIRE(lib::ImplicitlyDispatchConvertible<
                             const size_t&,
                             const size_t&>);
        CATCH_STATIC_REQUIRE(!lib::ImplicitlyDispatchConvertible<const size_t&, size_t&>);
        CATCH_STATIC_REQUIRE(!lib::ImplicitlyDispatchConvertible<const size_t&, size_t&&>);

        CATCH_STATIC_REQUIRE(lib::ImplicitlyDispatchConvertible<size_t&&, size_t>);
        CATCH_STATIC_REQUIRE(lib::ImplicitlyDispatchConvertible<size_t&&, const size_t&>);
        CATCH_STATIC_REQUIRE(!lib::ImplicitlyDispatchConvertible<size_t&&, size_t&>);
        CATCH_STATIC_REQUIRE(lib::ImplicitlyDispatchConvertible<size_t&&, size_t&&>);
    }

    CATCH_SECTION("Built in") {
        static_assert(lib::DispatchConvertible<size_t, size_t>);
        static_assert(!lib::DispatchConvertible<size_t, size_t&>);
        static_assert(lib::DispatchConvertible<size_t, const size_t&>);
        static_assert(lib::DispatchConvertible<size_t, size_t&&>);

        static_assert(!lib::DispatchConvertible<size_t&, size_t>);
        static_assert(lib::DispatchConvertible<size_t&, size_t&>);
        static_assert(lib::DispatchConvertible<size_t&, const size_t&>);
        static_assert(!lib::DispatchConvertible<size_t&, size_t&&>);

        static_assert(!lib::DispatchConvertible<const size_t&, size_t>);
        static_assert(!lib::DispatchConvertible<const size_t&, size_t&>);
        static_assert(lib::DispatchConvertible<const size_t&, const size_t&>);
        static_assert(!lib::DispatchConvertible<const size_t&, size_t&&>);

        static_assert(lib::DispatchConvertible<size_t&&, size_t>);
        static_assert(!lib::DispatchConvertible<size_t&&, size_t&>);
        static_assert(lib::DispatchConvertible<size_t&&, const size_t&>);
        static_assert(lib::DispatchConvertible<size_t&&, size_t&&>);
        //-------
        static_assert(lib::DispatchConvertible<lib::ExtentArg, lib::ExtentTag<10>>);

        CATCH_STATIC_REQUIRE(lib::dispatch_match<size_t, size_t>(0) == lib::perfect_match);
        CATCH_STATIC_REQUIRE(lib::dispatch_convert<size_t, size_t>(size_t(10)) == 10);
    }

    CATCH_SECTION("Simple Conversion 2.0") {
        auto tester = []<typename From, typename To>() {
            Uncopyable x = 10;
            const Uncopyable* ptr = &x;

            // This is the function that receives the converted argument as a type `To`.
            auto check_lambda = [ptr](To arg) {
                // Value must be preserved through this chain.
                CATCH_REQUIRE(arg.value_ == 10);

                // If we're going from a reference to a reference, then make sure we have
                // the same address as the original.
                if constexpr (std::is_reference_v<From> && std::is_reference_v<To>) {
                    CATCH_REQUIRE(&arg == ptr);
                } else {
                    CATCH_REQUIRE(&arg != ptr);
                }

                // If passed by mutable reference, mutate the argument to ensure this
                // mutation is visible from the caller.
                if constexpr (is_mutable_reference_v<From> && is_mutable_reference_v<To>) {
                    arg.value_ = 5;
                }
            };

            // Use `static_cast<From&&>` to correctly forward `x` in the case where we
            // pass by value.
            check_conversion_call<From, To>(static_cast<From&&>(x), check_lambda);

            // Check that we can observe mutation.
            if constexpr (is_mutable_reference_v<From> && is_mutable_reference_v<To>) {
                CATCH_REQUIRE(x.value_ == 5);
            }

            constexpr bool should_be_moved_from =
                !std::is_reference_v<From> || !std::is_reference_v<To>;

            if constexpr (should_be_moved_from) {
                fmt::print("From is reference: {}\n", std::is_reference_v<From>);
                fmt::print("To is reference: {}\n", std::is_reference_v<To>);
                CATCH_REQUIRE(x.is_moved_from());
                fmt::print("Here!\n");
            } else {
                CATCH_REQUIRE(!x.is_moved_from());
            }
        };

        TEST_ALL_COMBINATIONS(Uncopyable, Uncopyable, tester);
    }

    CATCH_SECTION("Variant") {
        CATCH_SECTION("Static Asserts") {
            using T = std::variant<uint64_t, int64_t>;

            // From as Value
            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<T, int64_t>);
            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<T, uint64_t>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T, float>);

            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<T, const int64_t&>);
            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<T, const uint64_t&>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T, const float&>);

            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T, int64_t&>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T, uint64_t&>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T, float&>);

            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<T, int64_t&&>);
            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<T, uint64_t&&>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T, float&&>);

            // From as ConstRef
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<const T&, int64_t>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<const T&, uint64_t>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<const T&, float>);

            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<const T&, const int64_t&>);
            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<const T&, const uint64_t&>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<const T&, const float&>);

            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<const T&, int64_t&>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<const T&, uint64_t&>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<const T&, float&>);

            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<const T&, int64_t&&>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<const T&, uint64_t&&>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<const T&, float&&>);

            // From as Ref
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T&, int64_t>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T&, uint64_t>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T&, float>);

            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<T&, const int64_t&>);
            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<T&, const uint64_t&>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T&, const float&>);

            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<T&, int64_t&>);
            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<T&, uint64_t&>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T&, float&>);

            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T&, int64_t&&>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T&, uint64_t&&>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T&, float&&>);

            // From as rvalue-ref
            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<T, int64_t>);
            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<T, uint64_t>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T, float>);

            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<T, const int64_t&>);
            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<T, const uint64_t&>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T, const float&>);

            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T, int64_t&>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T, uint64_t&>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T, float&>);

            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<T, int64_t&&>);
            CATCH_STATIC_REQUIRE(lib::DispatchConvertible<T, uint64_t&&>);
            CATCH_STATIC_REQUIRE(!lib::DispatchConvertible<T, float&&>);
        }

        CATCH_SECTION("Uncopyable variant with copyable alternative") {
            using Vec = std::vector<int>;
            using VarT = std::variant<Uncopyable, Vec>;

            auto tester = []<typename From, typename To>() {
                auto v = Vec({1, 2, 3});
                const auto* ptr = v.data();
                auto input = VarT(std::move(v));

                CATCH_REQUIRE(std::get<Vec>(input).data() == ptr);
                CATCH_REQUIRE(lib::dispatch_match<From, To>(input) == lib::perfect_match);
                CATCH_STATIC_REQUIRE(lib::DispatchConvertible<From, To>);

                auto check_lambda = [ptr](To arg) {
                    // Check the the value is preserved.
                    CATCH_REQUIRE(arg == Vec({1, 2, 3}));

                    // The pointer should also be preserved - no copies.
                    CATCH_REQUIRE(arg.data() == ptr);

                    // Passed by mutable reference - ensure we can mutate our argument
                    // and have this mutation be visible to the caller.
                    if constexpr (is_mutable_reference_v<From> && is_mutable_reference_v<To>) {
                        arg.push_back(4);
                    }
                };

                check_conversion_call<From, To>(static_cast<From&&>(input), check_lambda);

                // Check that we can observe mutation.
                if constexpr (is_mutable_reference_v<From> && is_mutable_reference_v<To>) {
                    CATCH_REQUIRE(std::get<Vec>(input) == Vec{1, 2, 3, 4});
                }
            };

            TEST_ALL_COMBINATIONS(VarT, Vec, tester);
        }

        CATCH_SECTION("Uncopyable variant with copyable alternative - matching") {
            // Make sure that dispatch conversion applies, but runtime matching fails.
            using Vec = std::vector<int>;
            using VarT = std::variant<Uncopyable, Vec>;

            auto tester = []<typename From, typename To>() {
                auto v = Vec({1, 2, 3});
                auto input = VarT(std::move(v));

                CATCH_STATIC_REQUIRE(lib::DispatchConvertible<From, To>);
                CATCH_REQUIRE(lib::dispatch_match<From, To>(input) == lib::invalid_match);
            };

            TEST_ALL_COMBINATIONS(VarT, Uncopyable, tester);
        }

        CATCH_SECTION("Uncopyable variant with uncopyable alternative") {
            using Vec = std::vector<int>;
            using VarT = std::variant<Uncopyable, Vec>;

            auto tester = []<typename From, typename To>() {
                auto input = VarT(std::in_place_type<Uncopyable>, 10);

                CATCH_REQUIRE(std::get<Uncopyable>(input).value_ == 10);
                CATCH_REQUIRE(!std::get<Uncopyable>(input).is_moved_from());
                CATCH_REQUIRE(lib::dispatch_match<From, To>(input) == lib::perfect_match);
                CATCH_STATIC_REQUIRE(lib::DispatchConvertible<From, To>);

                auto check_lambda = [](To arg) {
                    // Check the the value is preserved.
                    CATCH_REQUIRE(arg.value_ == 10);

                    // Passed by mutable reference - ensure we can mutate our argument
                    // and have this mutation be visible to the caller.
                    if constexpr (is_mutable_reference_v<From> && is_mutable_reference_v<To>) {
                        arg.value_ = 5;
                    }
                };

                check_conversion_call<From, To>(static_cast<From&&>(input), check_lambda);

                // Check that we can observe mutation.
                if constexpr (is_mutable_reference_v<From> && is_mutable_reference_v<To>) {
                    CATCH_REQUIRE(std::get<Uncopyable>(input).value_ == 5);
                }

                constexpr bool move_expected =
                    !std::is_reference_v<From> || !std::is_reference_v<To>;
                if constexpr (move_expected) {
                    CATCH_REQUIRE(std::get<Uncopyable>(input).is_moved_from());
                }
            };

            TEST_ALL_COMBINATIONS(VarT, Uncopyable, tester);
        }

        CATCH_SECTION("Uncopyable variant with uncopyable alternative - matching") {
            using Vec = std::vector<int>;
            using VarT = std::variant<Uncopyable, Vec>;

            auto tester = []<typename From, typename To>() {
                auto input = VarT(std::in_place_type<Uncopyable>, 10);

                CATCH_STATIC_REQUIRE(lib::DispatchConvertible<From, To>);
                CATCH_REQUIRE(lib::dispatch_match<From, To>(input) == lib::invalid_match);
            };

            TEST_ALL_COMBINATIONS(VarT, Vec, tester);
        }
    }

    CATCH_SECTION("Extent Checking") {
        static_assert(lib::DispatchConvertible<
                      lib::ExtentArg,
                      lib::ExtentTag<svs::Dynamic>>);
        static_assert(lib::DispatchConvertible<lib::ExtentArg, lib::ExtentTag<10>>);
        static_assert(lib::DispatchConvertible<lib::ExtentArg, lib::ExtentTag<20>>);

        // Match dynamic tags.
        auto arg = lib::ExtentArg(svs::Dynamic, false);
        CATCH_REQUIRE(
            lib::dispatch_match<lib::ExtentArg, lib::ExtentTag<svs::Dynamic>>(arg) ==
            lib::perfect_match
        );

        // When both are dynamic - the "force" field should have no relevance.
        arg.force_ = true;
        using DynamicTag = lib::ExtentTag<svs::Dynamic>;
        CATCH_REQUIRE(
            lib::dispatch_match<lib::ExtentArg, DynamicTag>(arg) == lib::perfect_match
        );

        // Prefer a static extent but allow dynamic.
        arg = {20, false};
        CATCH_REQUIRE(
            lib::dispatch_match<lib::ExtentArg, DynamicTag>(arg) == lib::imperfect_match
        );

        // Require static
        arg = {20, true};
        CATCH_REQUIRE(
            lib::dispatch_match<lib::ExtentArg, DynamicTag>(arg) == lib::invalid_match
        );

        // Match static tags.
        arg = {20, false};
        CATCH_REQUIRE(
            lib::dispatch_match<lib::ExtentArg, lib::ExtentTag<20>>(arg) ==
            lib::perfect_match
        );

        arg = {20, true};
        CATCH_REQUIRE(
            lib::dispatch_match<lib::ExtentArg, lib::ExtentTag<20>>(arg) ==
            lib::perfect_match
        );

        // Mismatch in extents should never match.
        arg = {10, false};
        CATCH_REQUIRE(
            lib::dispatch_match<lib::ExtentArg, lib::ExtentTag<20>>(arg) ==
            lib::invalid_match
        );

        arg = {10, true};
        CATCH_REQUIRE(
            lib::dispatch_match<lib::ExtentArg, lib::ExtentTag<20>>(arg) ==
            lib::invalid_match
        );
    }

    CATCH_SECTION("Extent Matching Through Variant") {
        using Variant = std::variant<size_t, lib::ExtentArg>;
        using DynamicTag = lib::ExtentTag<svs::Dynamic>;
        using E20 = lib::ExtentTag<20>;
        using E10 = lib::ExtentTag<10>;

        // size_t alternative.
        auto x = Variant{std::in_place_type<size_t>, 10};
        CATCH_REQUIRE(lib::dispatch_match<Variant, size_t>(x) == lib::perfect_match);
        CATCH_REQUIRE(lib::dispatch_match<Variant, DynamicTag>(x) == lib::invalid_match);
        CATCH_REQUIRE(lib::dispatch_match<Variant, E20>(x) == lib::invalid_match);
        CATCH_REQUIRE(lib::dispatch_match<Variant, E10>(x) == lib::invalid_match);

        // extent-arg alternative - allow fuzzy matching.
        x = lib::ExtentArg{10, false};
        CATCH_REQUIRE(lib::dispatch_match<Variant, size_t>(x) == lib::invalid_match);
        CATCH_REQUIRE(lib::dispatch_match<Variant, DynamicTag>(x) == lib::imperfect_match);
        CATCH_REQUIRE(lib::dispatch_match<Variant, E20>(x) == lib::invalid_match);
        CATCH_REQUIRE(lib::dispatch_match<Variant, E10>(x) == lib::perfect_match);

        x = lib::ExtentArg{10, true};
        CATCH_REQUIRE(lib::dispatch_match<Variant, size_t>(x) == lib::invalid_match);
        CATCH_REQUIRE(lib::dispatch_match<Variant, DynamicTag>(x) == lib::invalid_match);
        CATCH_REQUIRE(lib::dispatch_match<Variant, E20>(x) == lib::invalid_match);
        CATCH_REQUIRE(lib::dispatch_match<Variant, E10>(x) == lib::perfect_match);
    }

    CATCH_SECTION("Documentation") {
        CATCH_REQUIRE(lib::dispatch_description<size_t, size_t>() == "all values");
        CATCH_REQUIRE(
            lib::dispatch_description<std::variant<int64_t, uint64_t>, int64_t>() ==
            "all values -- (union alternative 0)"
        );
    }

    CATCH_SECTION("Documentation Table Generation") {
        auto dispatch_args =
            lib::detail::Signature<void(size_t, std::variant<int64_t, uint64_t>)>();
        auto target_args = lib::detail::Signature<void(size_t, int64_t)>();

        auto doc_table = lib::detail::make_descriptors(dispatch_args, target_args);

        CATCH_REQUIRE(doc_table[0]() == "all values");
        CATCH_REQUIRE(doc_table[1]() == "all values -- (union alternative 0)");
    }

    CATCH_SECTION("Matcher Function Pointer") {
        auto a = lib::ExtentArg(20, false);
        auto b = std::variant<Uncopyable, size_t>(std::in_place_type<size_t>, 10);
        auto c = 20;

        using dispatch_sig =
            Signature<void(lib::ExtentArg, std::variant<Uncopyable, size_t>, int)>;

        CATCH_SECTION("Matcher") {
            // Explanation of expected results:
            // 0. The ExtentArg is an imperfect match with `svs::Dynamic`.
            // 1. `size_t` matches the current alternative in the variant.
            // 2. passthrough types are always perfect matches.
            auto fptr = lib::detail::make_matcher(
                dispatch_sig{}, Signature<void(lib::ExtentTag<svs::Dynamic>, size_t, int)>()
            );
            auto ret = fptr(a, b, c);
            CATCH_REQUIRE(ret == std::array<int64_t, 3>{1, 0, 0});

            // Deduction.
            auto f = [](lib::ExtentTag<svs::Dynamic>, size_t, int) {};
            static_assert(std::is_same_v<
                          lib::detail::arg_signature_t<decltype(f)>,
                          Signature<void(lib::ExtentTag<svs::Dynamic>, size_t, int)>>);
        }

        {
            // 0. Mismatch on `ExtentTag`
            // 1. Mismatch on the current alternative in the variant.
            // 2. passthrough types are always perfect matches.
            auto fptr = lib::detail::make_matcher(
                dispatch_sig{}, Signature<void(lib::ExtentTag<2>, Uncopyable, int)>{}
            );
            auto ret = fptr(a, b, c);
            CATCH_REQUIRE(ret == std::array<int64_t, 3>{-1, -1, 0});

            // Deduction.
            auto f = [](lib::ExtentTag<2>, Uncopyable, int) {};
            static_assert(std::is_same_v<
                          lib::detail::arg_signature_t<decltype(f)>,
                          Signature<void(lib::ExtentTag<2>, Uncopyable, int)>>);
        }

        {
            // 0. Perfect match on `ExtentTag`
            // 1. Mismatch on the current alternative in the variant.
            // 2. passthrough types are always perfect matches.
            auto fptr = lib::detail::make_matcher(
                dispatch_sig{}, Signature<void(lib::ExtentTag<20>, Uncopyable, int)>{}
            );
            auto ret = fptr(a, b, c);
            CATCH_REQUIRE(ret == std::array<int64_t, 3>{0, -1, 0});

            // Deduction.
            auto f = [](lib::ExtentTag<20>, Uncopyable, int) {};
            static_assert(std::is_same_v<
                          lib::detail::arg_signature_t<decltype(f)>,
                          Signature<void(lib::ExtentTag<20>, Uncopyable, int)>>);
        }
    }

    CATCH_SECTION("Converter") {
        auto a = lib::ExtentArg(20, false);
        auto b = std::variant<Uncopyable, size_t>(std::in_place_type<Uncopyable>, 10);
        auto c = std::vector<int>{1, 2, 3};
        auto* cptr = c.data();

        CATCH_SECTION("All const-ref") {
            bool called = false;
            auto f = [cptr, &called](
                         lib::ExtentTag<svs::Dynamic>,
                         const Uncopyable& ib,
                         const std::vector<int>& ic
                     ) {
                called = true;
                CATCH_REQUIRE(ib.value_ == 10);
                CATCH_REQUIRE(ic == std::vector<int>{1, 2, 3});
                CATCH_REQUIRE(ic.data() == cptr);
                return 20;
            };

            using dispatch_sig = Signature<
                size_t(lib::ExtentArg, const std::variant<Uncopyable, size_t>&, const std::vector<int>&)>;

            auto wrapped = lib::detail::make_converter(dispatch_sig{}, f);

            CATCH_REQUIRE(!called);
            CATCH_REQUIRE(wrapped(a, b, c) == 20);
            CATCH_REQUIRE(called);
        }

        CATCH_SECTION("Moving Arguments") {
            bool called = false;
            auto f = [cptr, &called](
                         lib::ExtentTag<svs::Dynamic>, Uncopyable ib, std::vector<int> ic
                     ) {
                called = true;
                CATCH_REQUIRE(ib.value_ == 10);
                CATCH_REQUIRE(ic == std::vector<int>{1, 2, 3});
                CATCH_REQUIRE(ic.data() == cptr);
                return 20;
            };

            using dispatch_sig = Signature<
                size_t(lib::ExtentArg, std::variant<Uncopyable, size_t>, std::vector<int>)>;

            auto wrapped = lib::detail::make_converter(dispatch_sig{}, f);

            CATCH_REQUIRE(!called);
            CATCH_REQUIRE(wrapped(a, std::move(b), std::move(c)) == 20);
            CATCH_REQUIRE(called);
        }
    }

    CATCH_SECTION("Dispatch Target") {
        using VarT = std::variant<Uncopyable, size_t>;
        auto make_target = [](auto&& f) {
            return lib::DispatchTarget<size_t, lib::ExtentArg, VarT&, std::vector<int>&>(
                svs::lib::dispatcher_no_docs, SVS_FWD(f)
            );
        };

        auto make_match = [](int64_t x0, int64_t x1, int64_t x2) {
            return std::array<int64_t, 3>{x0, x1, x2};
        };

        auto c = std::vector<int>{1, 2, 3};

        CATCH_SECTION("Const-ref") {
            bool called = false;
            auto f = make_target([&](lib::ExtentTag<svs::Dynamic>,
                                     const Uncopyable& b,
                                     const std::vector<int>& i) {
                called = true;
                CATCH_REQUIRE(b.value_ == 10);
                CATCH_REQUIRE(i == std::vector<int>{1, 2, 3});
                return 5;
            });

            auto var = VarT{std::in_place_type<size_t>, 10};
            CATCH_REQUIRE(f.check_match(lib::ExtentArg(), var, c) == make_match(0, -1, 0));

            var = Uncopyable(10);
            CATCH_REQUIRE(f.check_match(lib::ExtentArg(5), var, c) == make_match(1, 0, 0));

            CATCH_REQUIRE(
                f.check_match(lib::ExtentArg(5, true), var, c) == make_match(-1, 0, 0)
            );

            // Do the call with a matching combination.
            auto ret = f.invoke(lib::ExtentArg(5), var, c);
            CATCH_REQUIRE(ret == 5);
            CATCH_REQUIRE(called);
        }

        CATCH_SECTION("By Ref") {
            bool called = false;
            auto f = make_target([&](lib::ExtentTag<16>, size_t& b, std::vector<int>& i) {
                called = true;
                CATCH_REQUIRE(b == 10);
                b = 20;
                CATCH_REQUIRE(i == std::vector<int>{1, 2, 3});
                i.push_back(4);
                return 5;
            });

            auto var = VarT{std::in_place_type<size_t>, 10};
            CATCH_REQUIRE(f.check_match(lib::ExtentArg(), var, c) == make_match(-1, 0, 0));

            var = Uncopyable(10);
            CATCH_REQUIRE(
                f.check_match(lib::ExtentArg(16), var, c) == make_match(0, -1, 0)
            );

            var = size_t(10);
            CATCH_REQUIRE(
                f.check_match(lib::ExtentArg(5, true), var, c) == make_match(-1, 0, 0)
            );

            // Do the call with a matching combination.
            auto ret = f.invoke(lib::ExtentArg(16), var, c);
            CATCH_REQUIRE(ret == 5);
            CATCH_REQUIRE(std::get<size_t>(var) == 20);
            CATCH_REQUIRE(c == std::vector<int>{1, 2, 3, 4});
            CATCH_REQUIRE(called);
        }
    }

    CATCH_SECTION("Dispatcher") {
        using VarT = std::variant<Uncopyable, size_t>;
        auto dispatcher =
            lib::Dispatcher<std::vector<int>, lib::ExtentArg, VarT&, std::vector<int>>();

        // Target 0
        dispatcher.register_target(
            lib::dispatcher_build_docs,
            [](lib::ExtentTag<svs::Dynamic>, size_t& x, const std::vector<int>& y) {
                auto y_copy = y;
                x = 0;
                y_copy.push_back(-1);
                return y_copy;
            }
        );

        // Target 1
        dispatcher.register_target(

            [](lib::ExtentTag<svs::Dynamic>, Uncopyable& x, const std::vector<int>& y) {
                CATCH_REQUIRE(x.value_ == 20);
                x.value_ = 5;
                return y;
            }
        );

        // Target 2 - build docs as well.
        dispatcher.register_target(
            lib::dispatcher_build_docs,
            [](lib::ExtentTag<20>, const Uncopyable& x, std::vector<int> y) {
                CATCH_REQUIRE(x.value_ == 20);
                y.push_back(5);
                return y;
            }
        );

        // Check docs.
        {
            // Method 1 has not documentation
            CATCH_REQUIRE(dispatcher.description(1, 0) == "unknown");
            CATCH_REQUIRE(dispatcher.description(1, 1) == "unknown");
            CATCH_REQUIRE(dispatcher.description(1, 2) == "unknown");

            // Method 0
            CATCH_REQUIRE(dispatcher.description(0, 0) == "any");
            auto desc0_1 = dispatcher.description(0, 1);
            CATCH_REQUIRE(desc0_1.find("all values") != std::string::npos);
            CATCH_REQUIRE(desc0_1.find("alternative 1") != std::string::npos);
            CATCH_REQUIRE(dispatcher.description(0, 2) == "all values");

            // Method 2
            CATCH_REQUIRE(dispatcher.description(2, 0) == "20");
            auto desc2_1 = dispatcher.description(2, 1);
            CATCH_REQUIRE(desc2_1.find("all values") != std::string::npos);
            CATCH_REQUIRE(desc2_1.find("alternative 0") != std::string::npos);
            CATCH_REQUIRE(dispatcher.description(2, 2) == "all values");

            // Bounds exceptions.
            CATCH_REQUIRE_THROWS_AS(dispatcher.description(0, 3), svs::ANNException);
            CATCH_REQUIRE_THROWS_AS(dispatcher.description(1, 3), svs::ANNException);
            CATCH_REQUIRE_THROWS_AS(dispatcher.description(2, 3), svs::ANNException);
            CATCH_REQUIRE_THROWS_AS(dispatcher.description(3, 0), svs::ANNException);
        }

        // Start checking matches.
        {
            VarT x{std::in_place_type<size_t>, 10};
            auto [i, match] = dispatcher.best_match(lib::ExtentArg{5, false}, x, {1, 2, 3});
            CATCH_REQUIRE(i.value() == 0);
            CATCH_REQUIRE(match == std::array<int64_t, 3>{1, 0, 0});

            auto ret = dispatcher.invoke(lib::ExtentArg{5, false}, x, {1, 2, 3});
            // Check return value and side-effects.
            CATCH_REQUIRE(ret == std::vector<int>{1, 2, 3, -1});
            CATCH_REQUIRE(std::get<size_t>(x) == 0);

            // If we try to force the extent - then we don't have a suitable match.
            auto m = dispatcher.best_match(lib::ExtentArg{5, true}, x, {1, 2, 3});
            CATCH_REQUIRE(!m.first.has_value());
            CATCH_REQUIRE_THROWS_AS(
                dispatcher.invoke(lib::ExtentArg{5, true}, x, {1, 2, 3}), svs::ANNException
            );
        }

        {
            VarT x{std::in_place_type<Uncopyable>, 20};
            auto v = std::vector<int>{1, 2, 3};

            // Check that the better-match is preferred over the first match.
            {
                auto [i, match] = dispatcher.best_match(lib::ExtentArg{20}, x, v);
                CATCH_REQUIRE(i.value() == 2);
                CATCH_REQUIRE(match == std::array<int64_t, 3>{0, 0, 0});
            }

            {
                auto [i, match] = dispatcher.best_match(lib::ExtentArg{40}, x, v);
                CATCH_REQUIRE(i.value() == 1);
                CATCH_REQUIRE(match == std::array<int64_t, 3>{1, 0, 0});
            }

            auto ret = dispatcher.invoke(lib::ExtentArg{40}, x, v);
            CATCH_REQUIRE(ret == std::vector<int>{1, 2, 3});
            CATCH_REQUIRE(std::get<Uncopyable>(x).value_ == 5); // side-effect.

            x = Uncopyable(20);

            // If we move the outside vector into the function, we should get the same
            // data at the very end.
            //
            // Make sure the vector has enough space so it doesn't reallocate when we
            // push.
            v.push_back(5);
            v.resize(v.size() - 1);
            auto* ptr = v.data();
            ret = dispatcher.invoke(lib::ExtentArg{20}, x, std::move(v));
            CATCH_REQUIRE(ret == std::vector<int>{1, 2, 3, 5});
            CATCH_REQUIRE(ret.data() == ptr);
        }
    }
}
