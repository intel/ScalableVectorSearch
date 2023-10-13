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

#include <array>
#include <cstdint>
#include <span>
#include <tuple>
#include <vector>

#include "svs/lib/array.h"
#include "svs/lib/static.h"
#include "svs/lib/tuples.h"
#include "svs/third-party/fmt.h"

#include "catch2/catch_test_macros.hpp"

namespace {

namespace detail {
template <bool B> using bool_type = std::integral_constant<bool, B>;
}

#define COPY_FIELDS()                                          \
    id = other.id;                                             \
    allocations = other.allocations;                           \
    bytes_allocated = other.bytes_allocated;                   \
    deallocations = other.deallocations;                       \
    bytes_deallocated = other.bytes_deallocated;               \
    default_constructors = other.default_constructors;         \
    copy_constructors = other.copy_constructors;               \
    move_constructors = other.move_constructors;               \
    destructors = other.destructors;                           \
    allocator_copy_construct = other.allocator_copy_construct; \
    allocator_copy_assign = other.allocator_copy_assign;       \
    allocator_move_construct = other.allocator_move_construct; \
    allocator_move_assign = other.allocator_move_assign;       \
    allocator_swaps = other.allocator_swaps

#define TEST_SHOW(x) fmt::print(#x " = {}\n", x)

template <typename T, bool C, bool M, bool S> struct TestAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U> struct rebind {
        using other = TestAllocator<U, C, M, S>;
    };

    using propagate_on_container_copy_assignment = detail::bool_type<C>;
    using propagate_on_container_move_assignment = detail::bool_type<M>;
    using propagate_on_container_swap = detail::bool_type<S>;

    constexpr TestAllocator() = default;
    constexpr TestAllocator(size_t id_)
        : id{id_} {}

    // Implementation - delegate to std::allocator
    using alloc = std::allocator<T>;
    using atraits = std::allocator_traits<alloc>;

    // Hooks
    [[nodiscard]] value_type* allocate(size_t n) {
        ++allocations;
        bytes_allocated += sizeof(T) * n;
        auto a = alloc();
        return atraits::allocate(a, n);
    }

    void deallocate(value_type* ptr, size_t count) {
        deallocations += 1;
        bytes_deallocated += sizeof(T) * count;
        auto a = alloc();
        atraits::deallocate(a, ptr, count);
    }

    template <typename U> void construct(U* p) {
        ++default_constructors;
        auto a = alloc();
        atraits::construct(a, p);
    }

    template <typename U> void construct(U* p, const T& x) {
        ++copy_constructors;
        auto a = alloc();
        atraits::construct(a, p, x);
    }

    template <typename U> void construct(U* p, T&& x) {
        ++move_constructors;
        auto a = alloc();
        atraits::construct(a, p, std::move(x));
    }

    template <typename U> void destroy(U* p) {
        ++destructors;
        auto a = alloc();
        atraits::destroy(a, p);
    }

    // Special member functions.
    TestAllocator(const TestAllocator& other) {
        COPY_FIELDS();
        ++allocator_copy_construct;
    }

    TestAllocator& operator=(const TestAllocator& other) {
        COPY_FIELDS();
        ++allocator_copy_assign;
        return *this;
    }

    TestAllocator(TestAllocator&& other) {
        COPY_FIELDS();
        ++allocator_move_construct;
    }

    TestAllocator& operator=(TestAllocator&& other) {
        COPY_FIELDS();
        ++allocator_move_assign;
        return *this;
    }

    void swap(TestAllocator& other) {
        auto tmp = *this;
        *this = other;
        other = tmp;
        // Correct for the number of copy constructors and assignments.
        --allocator_copy_assign;
        --other.allocator_copy_assign;
        --other.allocator_copy_construct;

        ++allocator_swaps;
        ++other.allocator_swaps;
    }

    friend void swap(TestAllocator& a, TestAllocator& b) { a.swap(b); }

    void show() const {
        TEST_SHOW(id);
        TEST_SHOW(allocations);
        TEST_SHOW(bytes_allocated);
        TEST_SHOW(deallocations);
        TEST_SHOW(bytes_deallocated);
        TEST_SHOW(copy_constructors);
        TEST_SHOW(move_constructors);
        TEST_SHOW(destructors);
        TEST_SHOW(allocator_copy_construct);
        TEST_SHOW(allocator_copy_assign);
        TEST_SHOW(allocator_move_construct);
        TEST_SHOW(allocator_move_assign);
        TEST_SHOW(allocator_swaps);
    }

  public:
    size_t id = 0;
    size_t allocations = 0;
    size_t bytes_allocated = 0;
    size_t deallocations = 0;
    size_t bytes_deallocated = 0;
    size_t default_constructors = 0;
    size_t copy_constructors = 0;
    size_t move_constructors = 0;
    size_t destructors = 0;

    // Container parameters.
    size_t allocator_copy_construct = 0;
    size_t allocator_copy_assign = 0;
    size_t allocator_move_construct = 0;
    size_t allocator_move_assign = 0;
    size_t allocator_swaps = 0;
};

template <typename T, typename U, bool C, bool M, bool S>
bool operator==(const TestAllocator<T, C, M, S>& a, const TestAllocator<U, C, M, S>& b) {
    return a.id == b.id;
}

using CheckType = std::array<size_t, 14>;

template <typename T, bool C, bool M, bool S>
void check_equal(const TestAllocator<T, C, M, S>& alloc, const CheckType& values) {
    CATCH_REQUIRE(alloc.id == values[0]);
    CATCH_REQUIRE(alloc.allocations == values[1]);
    CATCH_REQUIRE(alloc.bytes_allocated == values[2]);
    CATCH_REQUIRE(alloc.deallocations == values[3]);
    CATCH_REQUIRE(alloc.bytes_deallocated == values[4]);
    CATCH_REQUIRE(alloc.default_constructors == values[5]);
    CATCH_REQUIRE(alloc.copy_constructors == values[6]);
    CATCH_REQUIRE(alloc.move_constructors == values[7]);
    CATCH_REQUIRE(alloc.destructors == values[8]);
    CATCH_REQUIRE(alloc.allocator_copy_construct == values[9]);
    CATCH_REQUIRE(alloc.allocator_copy_assign == values[10]);
    CATCH_REQUIRE(alloc.allocator_move_construct == values[11]);
    CATCH_REQUIRE(alloc.allocator_move_assign == values[12]);
    CATCH_REQUIRE(alloc.allocator_swaps == values[13]);
}

///
/// Test that 1-dimensional indexing works correctly for static and dynamic
/// dimensions.
///
/// The argument can either be an integer or a `svs::meta::Val` type.
///
template <typename T> void test_offset_1d(T bound1) {
    // The internal representation of dimensions is as a heterogeneous tuple.
    // Here, that tuple is constructed.
    auto dims = std::make_tuple(bound1);

    // Convert the argument (whether it's an integer or a `Val`) to a `size_t`.
    size_t b1 = svs::meta::unwrap(bound1);
    size_t expected = 0;
    for (size_t i = 0; i < b1; ++i) {
        CATCH_REQUIRE(
            svs::detail::offset(dims, svs::detail::unchecked_make_array(i)) == expected
        );
        ++expected;
    }
    CATCH_REQUIRE(expected == b1);
}

template <typename T, typename U> void test_offset_2d(T bound1, U bound2) {
    auto dims = std::make_tuple(bound1, bound2);
    size_t b1 = svs::meta::unwrap(bound1);
    size_t b2 = svs::meta::unwrap(bound2);
    size_t upper = b1 * b2;

    size_t expected = 0;
    for (size_t j = 0; j < b1; ++j) {
        for (size_t i = 0; i < b2; ++i) {
            CATCH_REQUIRE(
                svs::detail::offset(dims, svs::detail::unchecked_make_array(j, i)) ==
                expected
            );
            ++expected;
        }
    }
    CATCH_REQUIRE(expected == upper);
}

template <typename T, typename U, typename V>
void test_offset_3d(T bound1, U bound2, V bound3) {
    auto dims = std::make_tuple(bound1, bound2, bound3);
    size_t b1 = svs::meta::unwrap(bound1);
    size_t b2 = svs::meta::unwrap(bound2);
    size_t b3 = svs::meta::unwrap(bound3);
    size_t upper = b1 * b2 * b3;

    size_t expected = 0;
    for (size_t k = 0; k < b1; ++k) {
        for (size_t j = 0; j < b2; ++j) {
            for (size_t i = 0; i < b3; ++i) {
                CATCH_REQUIRE(
                    svs::detail::offset(dims, svs::detail::unchecked_make_array(k, j, i)) ==
                    expected
                );
                ++expected;
            }
        }
    }
    CATCH_REQUIRE(expected == upper);
}
} // namespace

CATCH_TEST_CASE("Array Utilities", "[core][array]") {
    CATCH_SECTION("Make Array") {
        uint16_t a = 1;
        uint32_t b = 1000;
        size_t c = 1234;
        uint8_t d = 5;
        CATCH_REQUIRE(svs::detail::unchecked_make_array(a) == std::to_array<size_t>({a}));
        CATCH_REQUIRE(
            svs::detail::unchecked_make_array(a, b) == std::to_array<size_t>({a, b})
        );
        CATCH_REQUIRE(
            svs::detail::unchecked_make_array(a, b, c) == std::to_array<size_t>({a, b, c})
        );
        CATCH_REQUIRE(
            svs::detail::unchecked_make_array(a, b, c, d) ==
            std::to_array<size_t>({a, b, c, d})
        );
    }

    CATCH_SECTION("Default Strides", "[core][array]") {
        namespace dt = svs::detail;
        namespace meta = svs::meta;
        // Dimension 1
        {
            auto a = std::tuple<size_t>(100);
            auto b = std::make_tuple(meta::Val<1>{});
            CATCH_REQUIRE(dt::default_strides(a) == std::to_array<size_t>({1}));
            CATCH_REQUIRE(dt::default_strides(b) == std::to_array<size_t>({1}));
        }
        // Dimension 2
        {
            auto a = std::make_tuple(size_t{10}, size_t{100});
            auto b = std::make_tuple(meta::Val<10>{}, size_t{100});
            auto c = std::make_tuple(size_t{10}, meta::Val<100>{});
            auto d = std::make_tuple(meta::Val<10>{}, meta::Val<100>{});

            auto expected = std::to_array<size_t>({100, 1});
            CATCH_REQUIRE(dt::default_strides(a) == expected);
            CATCH_REQUIRE(dt::default_strides(b) == expected);
            CATCH_REQUIRE(dt::default_strides(c) == expected);
            CATCH_REQUIRE(dt::default_strides(d) == expected);
        }
    }

    CATCH_SECTION("Offset", "[core][array]") {
        namespace dt = svs::detail;
        namespace meta = svs::meta;
        // Dimension 1
        {
            test_offset_1d(size_t{128});
            test_offset_1d(meta::Val<128>{});
        }
        // Dimension 2
        {
            test_offset_2d(size_t{128}, size_t{10});
            test_offset_2d(meta::Val<128>{}, size_t{10});
            test_offset_2d(meta::Val<128>{}, meta::Val<10>{});
        }
        // Dimension 3
        {
            test_offset_3d(size_t{5}, size_t{3}, size_t{4});
            test_offset_3d(meta::Val<5>{}, size_t{3}, size_t{4});
            test_offset_3d(size_t{5}, meta::Val<3>{}, size_t{4});
            test_offset_3d(meta::Val<5>{}, meta::Val<3>{}, size_t{4});
            test_offset_3d(size_t{5}, size_t{3}, meta::Val<4>{});
            test_offset_3d(meta::Val<5>{}, size_t{3}, meta::Val<4>{});
            test_offset_3d(size_t{5}, meta::Val<3>{}, meta::Val<4>{});
            test_offset_3d(meta::Val<5>{}, meta::Val<3>{}, meta::Val<4>{});
        }
    }

    CATCH_SECTION("Extent Propagation", "[core][array]") {
        namespace dt = svs::detail;
        namespace meta = svs::meta;
        CATCH_REQUIRE(dt::get_extent_impl<size_t> == svs::Dynamic);
        CATCH_REQUIRE(dt::get_extent_impl<meta::Val<10>> == 10);
        CATCH_REQUIRE(dt::get_extent_impl<meta::Val<128>> == 128);

        // 1D case
        CATCH_REQUIRE(dt::getextent<std::tuple<size_t>> == svs::Dynamic);
        CATCH_REQUIRE(dt::getextent<std::tuple<meta::Val<23>>> == 23);

        // 2D case
        CATCH_REQUIRE(dt::getextent<std::tuple<size_t, size_t>> == svs::Dynamic);
        CATCH_REQUIRE(dt::getextent<std::tuple<meta::Val<23>, size_t>> == svs::Dynamic);
        CATCH_REQUIRE(dt::getextent<std::tuple<size_t, meta::Val<23>>> == 23);
    }
}

namespace {

const size_t ID_A = 0xc0ffee;
const size_t ID_B = 0xdeadbeef;
using CheckMap = std::unordered_map<std::string, CheckType>;

template <bool C, bool M, bool S> struct Bools {};

template <typename T, typename Dims, bool C, bool M, bool S>
svs::DenseArray<T, Dims, TestAllocator<T, C, M, S>>
make_source_array(const Dims& dims, size_t n_elements, size_t id, Bools<C, M, S>) {
    bool is_id = id == ID_A || id == ID_B;
    CATCH_REQUIRE(is_id);

    auto array = svs::DenseArray<T, Dims, TestAllocator<T, C, M, S>>(
        dims, TestAllocator<T, C, M, S>(id)
    );
    const auto& alloc = array.get_allocator();
    check_equal(
        alloc,
        std::array<size_t, 14>{
            id, 1, sizeof(T) * n_elements, 0, 0, n_elements, 0, 0, 0, 1, 0, 0, 0, 0}
    );

    // Assign the contents based on whether the id is A or B.
    int start = 0;
    int increment = id == ID_A ? 1 : -1;
    for (auto it = array.begin(), e = array.end(); it != e; ++it) {
        *it = start;
        start += increment;
    }

    return array;
}

// Case: false, false, false

template <typename T> CheckType expected_copy_construct(size_t n1) {
    size_t sT = sizeof(T);
    return {ID_A, 2, sT * (n1 + n1), 0, 0, n1, n1, 0, 0, 2, 0, 0, 0, 0};
}

template <typename T> CheckType expected_move_construct(size_t n1) {
    size_t sT = sizeof(T);
    return {ID_A, 1, sT * n1, 0, 0, n1, 0, 0, 0, 1, 0, 1, 0, 0};
}

template <typename T, bool Propagate> CheckType expected_copy_assign(size_t n1, size_t n2) {
    size_t sT = sizeof(T);
    if constexpr (Propagate) {
        return {ID_A, 2, sT * (n1 + n1), 0, 0, n1, n1, 0, 0, 1, 1, 0, 0, 0};
    } else {
        return {ID_A, 2, sT * (n2 + n1), 1, sT * n2, n2, n1, 0, n2, 1, 0, 0, 0, 0};
    }
}

template <typename T, bool Propagate>
CheckType expected_move_assign_eq(size_t n1, size_t n2) {
    size_t sT = sizeof(T);
    if constexpr (Propagate) {
        return {ID_A, 1, sT * n1, 0, 0, n1, 0, 0, 0, 1, 0, 0, 1, 0};
    } else {
        return {ID_A, 1, sT * n2, 1, sT * n2, n2, 0, 0, n2, 1, 0, 0, 0, 0};
    }
}

template <typename T, bool Propagate>
CheckType expected_move_assign_neq(size_t n1, size_t n2) {
    size_t sT = sizeof(T);
    if constexpr (Propagate) {
        return {ID_A, 1, sT * n1, 0, 0, n1, 0, 0, 0, 1, 0, 0, 1, 0};
    } else {
        return {ID_B, 2, sT * (n2 + n1), 1, sT * n2, n2, 0, n1, n2, 1, 0, 0, 0, 0};
    }
}

template <typename T, bool Propagate> CheckType expected_swap_eq1(size_t n1, size_t n2) {
    size_t sT = sizeof(T);
    if constexpr (Propagate) {
        return {ID_A, 1, sT * n2, 0, 0, n2, 0, 0, 0, 1, 0, 0, 0, 1};
    } else {
        return {ID_A, 1, sT * n1, 0, 0, n1, 0, 0, 0, 1, 0, 0, 0, 0};
    }
}

template <typename T, bool Propagate> CheckType expected_swap_eq2(size_t n1, size_t n2) {
    size_t sT = sizeof(T);
    if constexpr (Propagate) {
        return {ID_A, 1, sT * n1, 0, 0, n1, 0, 0, 0, 1, 0, 0, 0, 1};
    } else {
        return {ID_A, 1, sT * n2, 0, 0, n2, 0, 0, 0, 1, 0, 0, 0, 0};
    }
}

template <typename T> CheckType expected_swap_neq1(size_t SVS_UNUSED(n1), size_t n2) {
    size_t sT = sizeof(T);
    return {ID_B, 1, sT * n2, 0, 0, n2, 0, 0, 0, 1, 0, 0, 0, 1};
}

template <typename T> CheckType expected_swap_neq2(size_t n1, size_t SVS_UNUSED(n2)) {
    size_t sT = sizeof(T);
    return {ID_A, 1, sT * n1, 0, 0, n1, 0, 0, 0, 1, 0, 0, 0, 1};
}

template <typename T, typename Dims1, typename Dims2, bool C, bool M, bool S>
void test_array_allocators(
    const Dims1& dims1, size_t n1, const Dims2& dims2, size_t n2, Bools<C, M, S> b
    // const CheckMap& checker
) {
    auto make_array = [&](const auto& dims, size_t n, size_t id) {
        return make_source_array<T>(dims, n, id, b);
    };

    auto array_equal = [](const auto& a, const auto& b) {
        CATCH_REQUIRE(a.dims() == b.dims());
        CATCH_REQUIRE(std::equal(a.begin(), a.end(), b.begin()));
    };

    // Copy Construct.
    {
        fmt::print("Copy Construct\n");
        auto array = make_array(dims1, n1, ID_A);
        auto other = array;
        // other.get_allocator().show();
        check_equal(other.get_allocator(), expected_copy_construct<T>(n1));

        array_equal(other, array);
    }

    // Copy Assignment - equal allocators.
    {
        fmt::print("Copy Assign Equal\n");
        auto array = make_array(dims1, n1, ID_A);
        auto other = make_array(dims2, n2, ID_A);
        other = array;
        // other.get_allocator().show();
        check_equal(other.get_allocator(), expected_copy_assign<T, C>(n1, n2));
        array_equal(array, other);
    }

    // Move Construct.
    {
        fmt::print("Move Construct\n");
        auto array = make_array(dims1, n1, ID_A);
        auto other = std::move(array);
        // other.get_allocator().show();
        check_equal(other.get_allocator(), expected_move_construct<T>(n1));
        CATCH_REQUIRE(array.data() == nullptr);
        array_equal(other, make_array(dims1, n1, ID_A));
    }

    // Move Assignment - equal allocators.
    {
        fmt::print("Move Assign Equal\n");
        auto array = make_array(dims1, n1, ID_A);
        auto other = make_array(dims2, n2, ID_A);
        other = std::move(array);
        // other.get_allocator().show();
        check_equal(other.get_allocator(), expected_move_assign_eq<T, M>(n1, n2));

        CATCH_REQUIRE(array.data() == nullptr);
        array_equal(other, make_array(dims1, n1, ID_A));
    }

    // Move Assignment - unequal allocators.
    {
        fmt::print("Move Assign Unequal\n");
        auto array = make_array(dims1, n1, ID_A);
        auto other = make_array(dims2, n2, ID_B);
        other = std::move(array);
        // other.get_allocator().show();
        check_equal(other.get_allocator(), expected_move_assign_neq<T, M>(n1, n2));

        CATCH_REQUIRE(array.data() == nullptr);
        array_equal(other, make_array(dims1, n1, ID_A));
    }

    // Swap - equal allocators
    {
        using std::swap;
        fmt::print("Swaps Equal\n");
        auto array = make_array(dims1, n1, ID_A);
        auto other = make_array(dims2, n2, ID_A);
        swap(array, other);
        // other.get_allocator().show();
        fmt::print("Other\n");
        check_equal(other.get_allocator(), expected_swap_eq2<T, S>(n1, n2));
        fmt::print("Array\n");
        check_equal(array.get_allocator(), expected_swap_eq1<T, S>(n1, n2));
        array_equal(other, make_array(dims1, n1, ID_A));
        array_equal(array, make_array(dims2, n2, ID_A));
    }

    // Swap - unequal allocators.
    // Only applies if `propagate_on_container_swap` is true.
    if constexpr (S) {
        using std::swap;
        fmt::print("Swaps Unequal\n");
        auto array = make_array(dims1, n1, ID_A);
        auto other = make_array(dims2, n2, ID_B);
        swap(array, other);
        check_equal(other.get_allocator(), expected_swap_neq2<T>(n1, n2));
        check_equal(array.get_allocator(), expected_swap_neq1<T>(n1, n2));
        array_equal(other, make_array(dims1, n1, ID_A));
        array_equal(array, make_array(dims2, n2, ID_B));
    }
}

} // namespace

CATCH_TEST_CASE("Array", "[core][array_allocator]") {
    auto dims1 = std::tuple<size_t, size_t>(2, 3);
    auto n1 = 6;
    auto dims2 = std::tuple<size_t, size_t>(3, 4);
    auto n2 = 12;

    test_array_allocators<float>(dims1, n1, dims2, n2, Bools<false, false, false>{});
    test_array_allocators<float>(dims1, n1, dims2, n2, Bools<false, false, true>{});

    test_array_allocators<float>(dims1, n1, dims2, n2, Bools<false, true, false>{});
    test_array_allocators<float>(dims1, n1, dims2, n2, Bools<false, true, true>{});

    test_array_allocators<float>(dims1, n1, dims2, n2, Bools<true, false, false>{});
    test_array_allocators<float>(dims1, n1, dims2, n2, Bools<true, false, true>{});

    test_array_allocators<float>(dims1, n1, dims2, n2, Bools<true, true, false>{});
    test_array_allocators<float>(dims1, n1, dims2, n2, Bools<true, true, true>{});
}

CATCH_TEST_CASE("Array", "[core][array]") {
    std::vector<int> input{};
    for (size_t i = 0; i < 100; ++i) {
        input.push_back(i);
    }
    CATCH_SECTION("Square Dynamic") {
        auto x = svs::make_dense_array<int>(10, 10);
        std::copy(input.begin(), input.end(), x.begin());

        CATCH_REQUIRE(x.ndims() == 2);
        CATCH_REQUIRE(x.dims() == std::to_array<size_t>({10, 10}));
        CATCH_REQUIRE(x.getsize<0>() == 10);
        CATCH_REQUIRE(x.getsize<1>() == 10);

        // Views
        auto vx = x.view();
        CATCH_REQUIRE(x.dims() == vx.dims());
        CATCH_REQUIRE(std::equal(x.begin(), x.end(), vx.begin()));
        CATCH_REQUIRE(x.data() == vx.data());

        auto cvx = x.view();
        CATCH_REQUIRE(x.dims() == cvx.dims());
        CATCH_REQUIRE(std::equal(x.begin(), x.end(), cvx.begin()));
        CATCH_REQUIRE(x.data() == cvx.data());

        size_t count = 0;
        for (size_t j = 0; j < x.getsize<0>(); ++j) {
            auto slice = x.slice(j);
            CATCH_REQUIRE(
                std::equal(slice.begin(), slice.end(), input.begin() + j * x.getsize<1>())
            );

            for (size_t i = 0; i < x.getsize<1>(); ++i) {
                CATCH_REQUIRE(x.at(j, i) == input[count]);
                ++count;
                x.at(j, i) = 0;
            }
        }
        CATCH_REQUIRE(std::all_of(x.begin(), x.end(), [](auto i) { return i == 0; }));
        CATCH_REQUIRE(x.extent() == svs::Dynamic);
    }

    CATCH_SECTION("Rectangular Dynamic") {
        auto x = svs::make_dense_array<int>(4, 25);
        std::copy(input.begin(), input.end(), x.begin());

        // Views
        auto vx = x.view();
        CATCH_REQUIRE(x.dims() == vx.dims());
        CATCH_REQUIRE(std::equal(x.begin(), x.end(), vx.begin()));
        CATCH_REQUIRE(x.data() == vx.data());

        auto cvx = x.view();
        CATCH_REQUIRE(x.dims() == cvx.dims());
        CATCH_REQUIRE(std::equal(x.begin(), x.end(), cvx.begin()));
        CATCH_REQUIRE(x.data() == cvx.data());

        CATCH_REQUIRE(x.dims() == std::to_array<size_t>({4, 25}));
        CATCH_REQUIRE(x.getsize<0>() == 4);
        CATCH_REQUIRE(x.getsize<1>() == 25);
        size_t count = 0;
        for (size_t j = 0; j < x.getsize<0>(); ++j) {
            auto slice = x.slice(j);
            CATCH_REQUIRE(slice.size() == x.getsize<1>());
            CATCH_REQUIRE(
                std::equal(slice.begin(), slice.end(), input.begin() + j * x.getsize<1>())
            );

            for (size_t i = 0; i < x.getsize<1>(); ++i) {
                CATCH_REQUIRE(x.at(j, i) == input[count]);
                ++count;
                x.at(j, i) = 0;
            }
        }
        CATCH_REQUIRE(std::all_of(x.begin(), x.end(), [](auto i) { return i == 0; }));
        CATCH_REQUIRE(x.extent() == svs::Dynamic);
    }

    CATCH_SECTION("Rectangular Mixed") {
        constexpr size_t extent = 4;
        auto x = svs::make_dense_array<int>(25, svs::meta::Val<extent>{});
        std::copy(input.begin(), input.end(), x.begin());

        // Views
        auto vx = x.view();
        CATCH_REQUIRE(x.dims() == vx.dims());
        CATCH_REQUIRE(std::equal(x.begin(), x.end(), vx.begin()));
        CATCH_REQUIRE(x.data() == vx.data());

        auto cvx = x.view();
        CATCH_REQUIRE(x.dims() == cvx.dims());
        CATCH_REQUIRE(std::equal(x.begin(), x.end(), cvx.begin()));
        CATCH_REQUIRE(x.data() == cvx.data());

        CATCH_REQUIRE(x.dims() == std::to_array<size_t>({25, extent}));
        CATCH_REQUIRE(x.getsize<1>() == extent);
        CATCH_REQUIRE(x.getsize<0>() == 25);
        size_t count = 0;
        for (size_t j = 0; j < x.getsize<0>(); ++j) {
            auto slice = x.slice(j);
            CATCH_REQUIRE(slice.extent == extent);
            CATCH_REQUIRE(std::equal(
                input.begin() + j * x.getsize<1>(),
                input.begin() + (j + 1) * x.getsize<1>(),
                slice.begin()
            ));

            for (size_t i = 0; i < x.getsize<1>(); ++i) {
                CATCH_REQUIRE(x.at(j, i) == input[count]);
                ++count;
                x.at(j, i) = 0;
            }
        }
        CATCH_REQUIRE(std::all_of(x.begin(), x.end(), [](auto i) { return i == 0; }));
        CATCH_REQUIRE(x.extent() == extent);
    }
}
