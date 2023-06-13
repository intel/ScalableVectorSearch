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

#pragma once

#include "svs/lib/memory.h"
#include "svs/lib/meta.h"
#include "svs/lib/narrow.h"
#include "svs/lib/tuples.h"

#include <array>
#include <concepts>
#include <memory>
#include <span>
#include <tuple>
#include <type_traits>

namespace svs {
namespace detail {

///
/// Turn a collection of integer indices into a `std::array<size_t>`.
/// Performs checked narrowing conversion if necessary.
///
template <typename... Ts> std::array<size_t, sizeof...(Ts)> make_array(Ts... i) {
    return std::array<size_t, sizeof...(Ts)>({lib::narrow<size_t>(i)...});
}

///
/// Turn a collection of integer indices into a `std::array<size_t>`.
/// Narrowing conversions are not checked and may result in imprecise conversion.
///
template <typename... Ts> std::array<size_t, sizeof...(Ts)> unchecked_make_array(Ts... i) {
    return std::array<size_t, sizeof...(Ts)>({static_cast<size_t>(i)...});
}

// Given a tuple of dimensions, compute the default row-major strides for the
// dimensions.
template <typename... Ts>
std::array<size_t, sizeof...(Ts)> default_strides(const std::tuple<Ts...>& dims) {
    std::array<size_t, sizeof...(Ts)> result;
    size_t s{1};
    size_t index{sizeof...(Ts) - 1};
    lib::foreach_r(dims, [&s, &result, &index](auto&& i) {
        result[index] = s;
        s *= meta::unwrap(i);
        --index;
    });
    return result;
}

// N.B.: Marking `offset` as inline is needed to at least convince GCC to actually
// inline this function (which is something we definitely want).
template <typename... Ts>
inline SVS_FORCE_INLINE size_t
offset(const std::tuple<Ts...>& dims, std::array<size_t, sizeof...(Ts)>&& inds) {
    size_t offset{0};
    size_t stride{1};
    size_t index{sizeof...(Ts) - 1};
    lib::foreach_r(dims, [&offset, stride, index, inds](auto&& i) mutable {
        offset += stride * inds[index];
        stride *= meta::unwrap(i);
        --index;
    });
    return offset;
}

template <size_t N>
constexpr inline SVS_FORCE_INLINE size_t
offset(const std::array<size_t, N>& dims, const std::array<size_t, N>& inds) {
    size_t offset = 0;
    size_t stride = 1;
    for (size_t i = 0; i < N; ++i) {
        offset += stride * inds[N - i - 1];
        stride *= dims[N - i - 1];
    }
    return offset;
}

// All types showing up as paramters in the dimension tuple of the DenseArray should
// either be `size_t` or `Val<N>` for some `N`.
//
// Often, though, the integer dimensions may get passed as different kinds of integers.
//
// The helper struct `canonical_form` can be used to determine what a type should be
// converted to.
template <typename T> struct canonical_form {
    using type = size_t;
};
template <size_t N> struct canonical_form<meta::Val<N>> {
    using type = meta::Val<N>;
};

template <typename T>
inline constexpr bool is_dim_v = meta::is_val_type_v<T> || std::is_convertible_v<T, size_t>;

template <typename T>
concept IsDim = is_dim_v<T>;

// Compute the extent of a slice, depending on whether the fastest-changing dimension is
// statically sized or not.
template <typename T> inline constexpr size_t get_extent_impl = Dynamic;
template <auto N> inline constexpr size_t get_extent_impl<meta::Val<N>> = N;

template <size_t N> struct DimTypeHelper {
    using type = meta::Val<N>;
};

template <> struct DimTypeHelper<Dynamic> {
    using type = size_t;
};

// The idea behind `getextent` is that we peel off the last type in the parameter pack
// for the dimension tuple and use that to determine if it is dynamically sized (i.e.,
// the type is a `size_t`), or statically sized (its type is a `Val<N>`).
template <typename T> inline constexpr size_t getextent = Dynamic;
template <typename... Elements>
inline constexpr size_t getextent<std::tuple<Elements...>> =
    get_extent_impl<std::tuple_element_t<sizeof...(Elements) - 1, std::tuple<Elements...>>>;
} // namespace detail

// Convert size type arguments to either a `size_t` (if they were passed as an integer)
// or to a `Val<N>` if they were passed as a `Val`.
template <typename T> using canonical_form_t = typename detail::canonical_form<T>::type;

// Utilities for communicating static vs dynamic extent in higher-level data structures.
template <size_t N> using dim_type_t = typename detail::DimTypeHelper<N>::type;

#define SVS_ARRAY_FORWARD(...) std::forward<decltype(__VA_ARGS__)>(__VA_ARGS__)

///
/// @brief A N-dimensional array class supporting compile-time dimensionality.
///
/// @tparam T The value type of the array. Must be a trivial type.
///
template <typename T, typename Dims, typename Base = lib::DefaultStorage<T>>
class DenseArray {
  private:
    static_assert(std::is_trivial_v<T>);

  public:
    /// Definition: T
    using value_type = T;
    /// Definition: T*
    using pointer = T*;
    /// Definition: const T*
    using const_pointer = const T*;
    /// Definition: value_type&
    using reference = value_type&;
    //// Definition: const value_type&
    using const_reference = const value_type&;
    using span = std::span<T, detail::getextent<Dims>>;
    using const_span = std::span<const T, detail::getextent<Dims>>;
    using base_type = Base;

    // Storage based traits
    using allocator_type = lib::memory::allocator_type_t<Base>;

    /// @brief Return the extent of the span returned for `slice`.
    static constexpr size_t extent() { return detail::getextent<Dims>; };

    /// @brief The number of dimensions for the array.
    [[nodiscard]] static constexpr size_t ndims() { return std::tuple_size_v<Dims>; }
    /// @brief The dimensions of the array.
    [[nodiscard]] constexpr std::array<size_t, ndims()> dims() const {
        return std::apply(
            [](auto... args) { return std::array<size_t, ndims()>{meta::unwrap(args)...}; },
            dims_
        );
    }

    [[nodiscard]] constexpr Dims static_dims() const { return dims_; }

    /// @brief Return the total number of elements contained in the array.
    [[nodiscard]] constexpr size_t size() const {
        return std::apply([](auto... args) { return (meta::unwrap(args) * ...); }, dims_);
    }

    /// @brief Return the memory footprint of the array in bytes.
    [[nodiscard]] constexpr size_t bytes() const { return sizeof(T) * size(); }

    ///
    /// @brief Return the value of the `i`th dimension.
    ///
    /// @tparam i The dimensions to query. Must be in `[0, ndims()]`.
    ///
    template <size_t i> [[nodiscard]] size_t getsize() const {
        return meta::unwrap(std::get<i>(dims_));
    }

    ///
    /// @brief Return the extent (compiletime value) of the `i`th dimension.
    ///
    /// @tparam i The dimensions to query. Must be in `[0, ndims()]`.
    ///
    /// If the queried dimension is dynamically sized, returns ``svs::Dyanmic``.
    ///
    template <size_t i> [[nodiscard]] static constexpr size_t getextent() {
        return detail::get_extent_impl<std::tuple_element_t<i, Dims>>;
    }

    // Indexing
    [[nodiscard]] std::array<size_t, ndims()> strides() const {
        return detail::default_strides(dims_);
    }

    // Given `ndims()` indices, compute the linear offset from the base pointer for the
    // element pointed to by those indices.
    //
    // TODO: Add a bounds chekcing version and allow bounds checking to be a compile-time
    // option for debugging.
    template <typename... Is> [[nodiscard]] size_t offset(Is&&... indices) const {
        static_assert(sizeof...(indices) == ndims());
        return detail::offset(
            dims(), detail::unchecked_make_array(SVS_ARRAY_FORWARD(indices)...)
        );
    }

    ///
    /// @brief Access the specified element.
    ///
    /// @param indices The indices to access. Must satisfy ``sizeof...(indices) == ndims()``
    ///
    /// It is the callers responsibility to ensure that all indices are inbounds.
    ///
    template <typename... Is> reference at(Is&&... indices) {
        return *(data() + offset(SVS_ARRAY_FORWARD(indices)...));
    }

    /// @copydoc at()
    template <typename... Is> const_reference at(Is&&... indices) const {
        return *(data() + offset(SVS_ARRAY_FORWARD(indices)...));
    }

    /// @brief Return a const reference to the first element of the array.
    [[nodiscard]] constexpr const_reference first() const { return *(data()); }

    /// @brief Return a const reference to the last element of the array.
    [[nodiscard]] constexpr const_reference last() const { return *(data() + size() - 1); }

    ///
    /// @brief Obtain a `std::span` over the requested row.
    ///
    /// @param indices The indices specifying the row to access. Must satisfy
    ///     ``sizeof...(indices) == ndims() - 1``.
    ///
    /// The returned span will have the same extent as the last dimension of the array.
    ///
    template <typename... Is> [[nodiscard]] span slice(Is&&... indices) {
        size_t o = offset(SVS_ARRAY_FORWARD(indices)..., 0);
        return span{data() + o, getsize<ndims() - 1>()};
    }

    /// @copydoc slice()
    template <typename... Is> [[nodiscard]] const_span slice(Is&&... indices) const {
        size_t o = offset(SVS_ARRAY_FORWARD(indices)..., 0);
        return const_span{data() + o, getsize<ndims() - 1>()};
    }

    [[nodiscard]] pointer data() { return lib::memory::access_storage(base_); }
    [[nodiscard]] const_pointer data() const { return lib::memory::access_storage(base_); }

    [[nodiscard]] const Base& getbase() const { return base_; }
    [[nodiscard]] Base& getbase_mutable() { return base_; }
    [[nodiscard]] Base&& acquire_base() { return std::move(base_); }

    // Special Members
    DenseArray() = default;

    // Copy constructor for owning data.
    // Creates another array from the same memory source and copies over the contents.
    DenseArray(const DenseArray& other)
        requires lib::memory::implicit_copy_enabled_v<base_type> &&
                     lib::memory::is_owning_v<base_type>
        : base_{lib::allocate_managed<T>(allocator_type{}, other.size())}
        , dims_{other.dims_} {
        std::copy(other.begin(), other.end(), begin());
    }

    DenseArray& operator=(const DenseArray& other)
        requires lib::memory::implicit_copy_enabled_v<base_type> &&
                 lib::memory::is_owning_v<base_type>
    {
        if (this != &other) {
            base_ = lib::allocate_managed<T>(allocator_type{}, other.size());
            dims_ = other.dims_;
            std::copy(other.begin(), other.end(), begin());
        }
        return *this;
    }

    // Non-owning implicit copy.
    DenseArray(const DenseArray& other)
        requires lib::memory::implicit_copy_enabled_v<base_type> &&
                     (!lib::memory::is_owning_v<base_type>)
        : base_{other.base_}
        , dims_{other.dims_} {}

    DenseArray& operator=(const DenseArray& other)
        requires lib::memory::implicit_copy_enabled_v<base_type> &&
                 (!lib::memory::is_owning_v<base_type>)
    {
        if (this != &other) {
            base_ = other.base_;
            dims_ = other.dims_;
        }
        return *this;
    }

    // TODO: Mark as `noexcept` depeding on the base.
    DenseArray(DenseArray&& /*unused*/) noexcept = default;
    DenseArray& operator=(DenseArray&& /*unused*/) noexcept = default;
    ~DenseArray() = default;

    /////
    ///// Constructors
    /////

    // If the base can be copied, we can use this constructor.
    explicit DenseArray(Base&& base, Dims dims)
        : base_{std::move(base)}
        , dims_{std::move(dims)} {}

    template <detail::IsDim... Ts>
    explicit DenseArray(const Base& base, Ts... dims)
        : base_{base}
        , dims_{Dims(dims...)} {}

    // If the base cannot be copied, but can only be moved, try to use this constructor.
    template <detail::IsDim... Ts>
    explicit DenseArray(Base&& base, Ts... dims)
        : base_{std::move(base)}
        , dims_{Dims(dims...)} {}

    // Construct from scratch
    // clang-format off
    template <detail::IsDim... Ts>
    explicit DenseArray(Ts... dims)
        requires lib::memory::may_trivially_construct<allocator_type>
        : base_{lib::allocate_managed<T>(allocator_type{}, (meta::unwrap(dims) * ...))}
        , dims_{Dims(dims...)} {}

    // Pass in an allocator to both allocate and configure dims.
    template <typename Allocator, detail::IsDim... Ts>
    explicit DenseArray(Allocator&& allocator, Ts... dims)
        requires lib::memory::is_allocator_v<std::decay_t<Allocator>>
        : base_{lib::allocate_managed<T>(
            SVS_ARRAY_FORWARD(allocator), (meta::unwrap(dims) * ...)
        )}
        , dims_{Dims(dims...)} {}
    // clang-format on

    // Iterator

    /// @brief Return a random access iterator to the beginning of the array.
    pointer begin() { return data(); }
    /// @brief Return a random access iterator to the beginning of the array.
    const_pointer begin() const { return data(); }
    /// @brief Return A random access iterator to the end of the array.
    pointer end() { return data() + size(); }
    /// @brief Return A random access iterator to the end of the array.
    const_pointer end() const { return data() + size(); }

    /// @brief Return a mutable view over the memory of this array.
    DenseArray<T, Dims, T*> view() { return DenseArray<T, Dims, T*>(begin(), dims_); }
    /// @brief Return a constant view over the memory of this array.
    DenseArray<const T, Dims, const T*> cview() const {
        return DenseArray<const T, Dims, const T*>(begin(), dims_);
    }
    /// @brief Return a constant view over the memory of this array.
    DenseArray<const T, Dims, const T*> view() const { return cview(); }

  private:
    Base base_;
    Dims dims_;
};

#undef SVS_ARRAY_FORWARD

template <size_t I, typename T, typename Dims, typename Base>
size_t getsize(const DenseArray<T, Dims, Base>& array) {
    return array.template getsize<I>();
}

template <size_t I, typename T, typename Dims, typename Base>
constexpr size_t getextent(const DenseArray<T, Dims, Base>& /*unused*/) {
    return DenseArray<T, Dims, Base>::template getextent<I>();
}

/////
///// make_dense_array
/////

///
/// @defgroup make_dense_array_family Constructors for ``svs::DenseArray``.
///

///
/// @class hidden_make_dense_array_dims
/// The number of dimensions is inferred from the number of arguments.
///
/// Each argument must be either convertible to ``size_t`` or a ``svs::lib::meta::Val`.
/// In the latter case, the corresponding dimension of the result array will be static.
///

///
/// @ingroup make_dense_array_family
/// @brief Construct an uninitialized ``DenseArray``.
///
/// @param dims The (potentially static) dimensions of the resulting array.
///
/// @copydoc hidden_make_dense_array_dims
///
template <typename T, detail::IsDim... Dims> auto make_dense_array(Dims... dims) {
    return DenseArray<T, std::tuple<canonical_form_t<Dims>...>, lib::DefaultStorage<T>>(
        dims...
    );
}

// clang-format off
// N.B.: Clang format doesn't seem to do a great job of formatting requires clauses.

///
/// @ingroup make_dense_array_family
/// @brief Construct a ``DenseArray`` using the given allocator.
///
/// @param allocator The allocator to use for memory.
/// @param dims The (potentially static) dimensions of the resulting array.
///
/// @copydoc hidden_make_dense_array_dims
///
template <typename T, typename Allocator, detail::IsDim... Dims>
requires lib::memory::is_allocator_v<std::decay_t<Allocator>>
auto make_dense_array(Allocator&& allocator, Dims... dims) {
    // Compute the storage type
    using storage_type = decltype(lib::allocate_managed<T>(
        std::forward<Allocator>(allocator), (meta::unwrap(dims) * ...)
    ));
    return DenseArray<T, std::tuple<canonical_form_t<Dims>...>, storage_type>(
        std::forward<Allocator>(allocator), dims...
    );
}

///
/// @ingroup make_dense_array_family
/// @brief Construct a ``DenseArray`` around the provided base object.
///
/// @param base The (smart) pointer owning the storage that is being used to back the array.
/// @param dims The (potentially static) dimensions of the resulting array.
///
/// @copydoc hidden_make_dense_array_dims
///
/// **NOTE**: If the deduced type for ``base`` is a (const qualified) pointer, the returned
/// array will be a non-owning view over the memory.
///
/// It is the caller's responsibility to ensure that sufficient memory has been allocated.
///
template <lib::memory::Storage Base, detail::IsDim... Dims>
auto make_dense_array(Base base, Dims... dims) {
    using T = lib::memory::storage_value_type_t<Base>;
    return DenseArray<T, std::tuple<canonical_form_t<Dims>...>, std::decay_t<Base>>(
        std::forward<Base>(base), dims...
    );
}

template <lib::memory::Storage Base, detail::IsDim... Dims>
auto make_dense_array(Base base, std::tuple<Dims...>&& dims) {
    // The idea here is that we move-capture `base` into the lambda (need to mark the
    // lambda as "mutable" because we're going to move the captured base out of the
    // lambda and into the new DenseArray).
    //
    // We then use `std::apply` to splat the tuple into the lambda
    auto fn = [inner = std::move(base)](auto... args) mutable {
        return make_dense_array(std::move(inner), args...);
    };
    return std::apply(fn, std::forward<decltype(dims)>(dims));
}
// clang-format on

/////
///// Type Aliases
/////

/// @brief Dynamically sized, unresizeable vector.
template <typename T> using Vector = DenseArray<T, std::tuple<size_t>>;

/// @brief Dynamically sized matrix.
template <typename T> using Matrix = DenseArray<T, std::tuple<size_t, size_t>>;

/// @brief Dynamically sized matrix view.
template <typename T> using MatrixView = DenseArray<T, std::tuple<size_t, size_t>, T*>;

/// @brief Dynamically sized constant matrix view.
template <typename T>
using ConstMatrixView = DenseArray<const T, std::tuple<size_t, size_t>, const T*>;

} // namespace svs
