/*
 * Copyright 2023 Intel Corporation
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
        s *= lib::as_integral(i);
        --index;
    });
    return result;
}

// N.B.: Marking `offset` as inline is needed to at least convince GCC to actually
// inline this function (which is something we definitely want).
template <typename... Ts>
SVS_FORCE_INLINE size_t
offset(const std::tuple<Ts...>& dims, std::array<size_t, sizeof...(Ts)>&& inds) {
    size_t offset{0};
    size_t stride{1};
    size_t index{sizeof...(Ts) - 1};
    lib::foreach_r(dims, [&offset, stride, index, inds](auto&& i) mutable {
        offset += stride * inds[index];
        stride *= lib::as_integral(i);
        --index;
    });
    return offset;
}

template <size_t N>
constexpr SVS_FORCE_INLINE size_t
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
template <size_t N> struct canonical_form<lib::Val<N>> {
    using type = lib::Val<N>;
};

template <typename T>
inline constexpr bool is_dim_v = lib::is_val_type_v<T> || std::is_convertible_v<T, size_t>;

template <typename T>
concept IsDim = is_dim_v<T>;

// Compute the extent of a slice, depending on whether the fastest-changing dimension is
// statically sized or not.
template <typename T> inline constexpr size_t get_extent_impl = Dynamic;
template <auto N> inline constexpr size_t get_extent_impl<lib::Val<N>> = N;

template <size_t N> struct DimTypeHelper {
    using type = lib::Val<N>;
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

template <detail::IsDim... Ts> std::tuple<canonical_form_t<Ts>...> make_dims(Ts&&... ts) {
    return std::tuple<canonical_form_t<Ts>...>(SVS_FWD(ts)...);
}

template <typename T> struct View {
  public:
    View(T* ptr_)
        : ptr{ptr_} {}

  public:
    T* ptr;
};

template <typename T> inline constexpr bool is_view_type_v = false;
template <typename T> inline constexpr bool is_view_type_v<View<T>> = true;

namespace array_impl {

// Shared implementations across various DenseArray specializations.
template <typename Dims> [[nodiscard]] constexpr size_t extent() {
    return svs::detail::getextent<Dims>;
}

template <typename Dims> [[nodiscard]] constexpr size_t ndims() {
    return std::tuple_size_v<Dims>;
}

template <typename Dims>
[[nodiscard]] constexpr SVS_FORCE_INLINE std::array<size_t, ndims<Dims>()>
dims(const Dims& dims) {
    return std::apply(
        [](auto... args) {
            return std::array<size_t, ndims<Dims>()>{lib::as_integral(args)...};
        },
        dims
    );
}

template <typename Dims>
[[nodiscard]] SVS_FORCE_INLINE constexpr size_t size(const Dims& dims) {
    return std::apply([](auto... args) { return (lib::as_integral(args) * ...); }, dims);
}

template <size_t i, typename Dims>
[[nodiscard]] SVS_FORCE_INLINE constexpr size_t getsize(const Dims& dims) {
    return lib::as_integral(std::get<i>(dims));
}

template <size_t i, typename Dims>
[[nodiscard]] SVS_FORCE_INLINE constexpr size_t getextent() {
    return svs::detail::get_extent_impl<std::tuple_element_t<i, Dims>>;
}

template <typename Dims>
[[nodiscard]] SVS_FORCE_INLINE std::array<size_t, ndims<Dims>()> strides(const Dims& dims) {
    return svs::detail::default_strides(dims);
}

template <typename Dims, typename... Is>
[[nodiscard]] SVS_FORCE_INLINE size_t offset(const Dims& dims, Is&&... indices) {
    static_assert(sizeof...(indices) == ndims<Dims>());
    return detail::offset(dims, detail::unchecked_make_array(SVS_FWD(indices)...));
}

} // namespace array_impl

///
/// @brief A N-dimensional array class supporting compile-time dimensionality.
///
/// @tparam T The value type of the array. Must be a trivial type.
///
template <typename T, typename Dims, typename Alloc = lib::Allocator<T>> class DenseArray {
  private:
    // N.B.: This is an important assumption for many algorithms of this type.
    // Don't remove this requirement without careful consideration.
    static_assert(std::is_trivial_v<T>);

  public:
    ///// Allocator Aware
    using allocator_type = Alloc;
    using atraits = std::allocator_traits<allocator_type>;
    using pointer = typename atraits::pointer;
    using const_pointer = typename atraits::const_pointer;

    ///// Container
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    using size_type = size_t;

    // Ensure consistency.
    static_assert(std::is_same_v<value_type, typename atraits::value_type>);

    ///// Misc Type Defs
    using const_span = std::span<const T, detail::getextent<Dims>>;
    using span = std::span<T, detail::getextent<Dims>>;

    // Get the underlying allocator.
    const allocator_type& get_allocator() const { return allocator_; }

    /// @brief Return the extent of the span returned for `slice`.
    static constexpr size_t extent() { return array_impl::extent<Dims>(); }

    /// @brief The number of dimensions for the array.
    [[nodiscard]] static constexpr size_t ndims() { return array_impl::ndims<Dims>(); }
    /// @brief The dimensions of the array.
    [[nodiscard]] constexpr std::array<size_t, ndims()> dims() const {
        return array_impl::dims(dims_);
    }

    [[nodiscard]] constexpr Dims static_dims() const { return dims_; }

    /// @brief Return the total number of elements contained in the array.
    [[nodiscard]] constexpr size_t size() const { return array_impl::size(dims_); }

    /// @brief Return the memory footprint of the array in bytes.
    [[nodiscard]] constexpr size_t bytes() const { return sizeof(T) * size(); }

    ///
    /// @brief Return the value of the `i`th dimension.
    ///
    /// @tparam i The dimensions to query. Must be in `[0, ndims()]`.
    ///
    template <size_t i> [[nodiscard]] size_t getsize() const {
        return array_impl::getsize<i>(dims_);
    }

    ///
    /// @brief Return the extent (compiletime value) of the `i`th dimension.
    ///
    /// @tparam i The dimensions to query. Must be in `[0, ndims()]`.
    ///
    /// If the queried dimension is dynamically sized, returns ``svs::Dyanmic``.
    ///
    template <size_t i> [[nodiscard]] static constexpr size_t getextent() {
        return array_impl::getextent<i, Dims>();
    }

    // Indexing
    [[nodiscard]] std::array<size_t, ndims()> strides() const {
        return array_impl::strides(dims_);
    }

    // Given `ndims()` indices, compute the linear offset from the base pointer for the
    // element pointed to by those indices.
    //
    // TODO: Add a bounds chekcing version and allow bounds checking to be a compile-time
    // option for debugging.
    template <typename... Is> [[nodiscard]] size_t offset(Is&&... indices) const {
        return array_impl::offset(dims_, SVS_FWD(indices)...);
    }

    ///
    /// @brief Access the specified element.
    ///
    /// @param indices The indices to access. Must satisfy ``sizeof...(indices) == ndims()``
    ///
    /// It is the callers responsibility to ensure that all indices are inbounds.
    ///
    template <typename... Is> reference at(Is&&... indices) {
        return *(data() + offset(SVS_FWD(indices)...));
    }

    /// @copydoc at()
    template <typename... Is> const_reference at(Is&&... indices) const {
        return *(data() + offset(SVS_FWD(indices)...));
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
        size_t o = offset(SVS_FWD(indices)..., 0);
        return span{data() + o, getsize<ndims() - 1>()};
    }

    /// @copydoc slice()
    template <typename... Is> [[nodiscard]] const_span slice(Is&&... indices) const {
        size_t o = offset(SVS_FWD(indices)..., 0);
        return const_span{data() + o, getsize<ndims() - 1>()};
    }

    [[nodiscard]] pointer data() { return pointer_; }
    [[nodiscard]] const_pointer data() const { return pointer_; }

    // Special Members
    DenseArray() = default;

    // Copy constructor for owning data.
    // Creates another array from the same memory source and copies over the contents.
    DenseArray(const DenseArray& other)
        : pointer_{nullptr}
        , dims_{other.dims_}
        , allocator_{
              atraits::select_on_container_copy_construction(other.get_allocator())} {
        size_t sz = other.size();
        pointer_ = atraits::allocate(allocator_, sz);
        assign(other.begin(), other.end());
    }

    DenseArray& operator=(const DenseArray& other) {
        if (this != &other) {
            // Because this container does not implement dynamic resizing, we always need
            // to tear down the current instance before copying over elements.
            //
            // We *could* add an optimization where both the source and the destination
            // are exactly the same size.
            if (pointer_ != nullptr) {
                tear_down();
            }
            // Conditionally propagate the other's allocator.
            if constexpr (atraits::propagate_on_container_copy_assignment::value) {
                allocator_ = other.allocator_;
            }
            // Copy over the dimensions.
            dims_ = other.dims_;
            size_t sz = size();
            // Allocate and copy contents.
            pointer_ = atraits::allocate(allocator_, sz);
            assign(other.begin(), other.end());
        }
        return *this;
    }

    DenseArray(DenseArray&& other) noexcept
        : pointer_{std::exchange(other.pointer_, nullptr)}
        , dims_{other.dims_}
        , allocator_{std::move(other.allocator_)} {}

    DenseArray& operator=(DenseArray&& other) {
        // Handle de-allocation of our current resoucres.
        if (pointer_ != nullptr) {
            tear_down();
        }

        if constexpr (atraits::propagate_on_container_move_assignment::value) {
            move_assign_pilfer(other);
        } else {
            move_assign_copy_if_unequal(other);
        }
        return *this;
    }

    ~DenseArray() noexcept {
        if (pointer_ != nullptr) {
            tear_down();
        }
    }

    void swap(DenseArray& other) {
        using std::swap;
        swap(pointer_, other.pointer_);
        swap(dims_, other.dims_);
        if constexpr (atraits::propagate_on_container_swap::value) {
            swap(allocator_, other.allocator_);
        }
    }

    // Define for ADL purposes.
    friend void swap(DenseArray& a, DenseArray& b) { a.swap(b); }

    /////
    ///// Constructors
    /////

    explicit DenseArray(Dims dims, const Alloc& allocator)
        : pointer_{nullptr}
        , dims_{std::move(dims)}
        , allocator_{allocator} {
        size_t sz = size();
        pointer_ = atraits::allocate(allocator_, sz);
        for (pointer p = pointer_, e = pointer_ + sz; p != e; ++p) {
            atraits::construct(allocator_, std::to_address(p));
        }
    }

    explicit DenseArray(Dims dims)
        : DenseArray(std::move(dims), Alloc()) {}

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
    DenseArray<T, Dims, View<T>> view() {
        return DenseArray<T, Dims, View<T>>(dims_, pointer_);
    }
    /// @brief Return a constant view over the memory of this array.
    DenseArray<const T, Dims, View<const T>> cview() const {
        return DenseArray<const T, Dims, View<const T>>(dims_, pointer_);
    }
    /// @brief Return a constant view over the memory of this array.
    DenseArray<const T, Dims, View<const T>> view() const { return cview(); }

  private:
    void tear_down() noexcept {
        size_t sz = size();
        for (pointer p = pointer_, e = pointer_ + sz; p != e; ++p) {
            atraits::destroy(allocator_, std::to_address(p));
        }
        atraits::deallocate(allocator_, pointer_, sz);
        pointer_ = nullptr;
    }

    template <typename It> void assign(It b, It e) {
        assert(std::distance(b, e) == std::distance(begin(), end()));
        assert(std::distance(b, e) == lib::narrow<int64_t>(size()));
        pointer p = pointer_;
        for (; b != e; ++b) {
            atraits::construct(allocator_, p, *b);
            ++p;
        }
    }

    // Move assignment by pilfering the pointer from the other array.
    // If `propagate_on_container_move_assignment`, then the allocate will also be move
    // assigned.
    void move_assign_pilfer(DenseArray& other) {
        if constexpr (atraits::propagate_on_container_move_assignment::value) {
            allocator_ = std::move(other.allocator_);
        }
        dims_ = std::move(other.dims_);
        pointer_ = std::exchange(other.pointer_, nullptr);
    }

    void move_assign_copy_if_unequal(DenseArray& other) {
        // If the two allocators are equal - then we can at least pilfer the pointer of the
        // other container.
        if (allocator_ == other.allocator_) {
            move_assign_pilfer(other);
            return;
        }

        dims_ = other.dims_;
        size_t sz = size();
        pointer_ = atraits::allocate(allocator_, sz);
        assign(std::move_iterator(other.begin()), std::move_iterator(other.end()));
        other.tear_down();
    }

  private:
    pointer pointer_{nullptr};
    [[no_unique_address]] Dims dims_{};
    [[no_unique_address]] Alloc allocator_;
};

template <typename T, typename Dims> class DenseArray<T, Dims, View<T>> {
  private:
    // N.B.: This is an important assumption for many algorithms of this type.
    // Don't remove this requirement without careful consideration.
    static_assert(std::is_trivial_v<T>);

  public:
    static constexpr bool is_const = std::is_const_v<T>;

    ///// Allocator Aware
    using pointer = T*;
    using const_pointer = const T*;

    ///// Container
    using value_type = std::remove_cv_t<T>;
    using reference = T&;
    using const_reference = const T&;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    using size_type = size_t;

    ///// Misc Type Defs
    using const_span = std::span<const T, detail::getextent<Dims>>;
    using span = std::span<T, detail::getextent<Dims>>;

    /// @brief Return the extent of the span returned for `slice`.
    static constexpr size_t extent() { return array_impl::extent<Dims>(); }

    /// @brief The number of dimensions for the array.
    [[nodiscard]] static constexpr size_t ndims() { return array_impl::ndims<Dims>(); }
    /// @brief The dimensions of the array.
    [[nodiscard]] constexpr std::array<size_t, ndims()> dims() const {
        return array_impl::dims(dims_);
    }

    [[nodiscard]] constexpr Dims static_dims() const { return dims_; }

    /// @brief Return the total number of elements contained in the array.
    [[nodiscard]] constexpr size_t size() const { return array_impl::size(dims_); }

    /// @brief Return the memory footprint of the array in bytes.
    [[nodiscard]] constexpr size_t bytes() const { return sizeof(T) * size(); }

    ///
    /// @brief Return the value of the `i`th dimension.
    ///
    /// @tparam i The dimensions to query. Must be in `[0, ndims()]`.
    ///
    template <size_t i> [[nodiscard]] size_t getsize() const {
        return array_impl::getsize<i>(dims_);
    }

    ///
    /// @brief Return the extent (compiletime value) of the `i`th dimension.
    ///
    /// @tparam i The dimensions to query. Must be in `[0, ndims()]`.
    ///
    /// If the queried dimension is dynamically sized, returns ``svs::Dyanmic``.
    ///
    template <size_t i> [[nodiscard]] static constexpr size_t getextent() {
        return array_impl::getextent<i, Dims>();
    }

    // Indexing
    [[nodiscard]] std::array<size_t, ndims()> strides() const {
        return array_impl::strides(dims_);
    }

    // Given `ndims()` indices, compute the linear offset from the base pointer for the
    // element pointed to by those indices.
    //
    // TODO: Add a bounds chekcing version and allow bounds checking to be a compile-time
    // option for debugging.
    template <typename... Is> [[nodiscard]] size_t offset(Is&&... indices) const {
        return array_impl::offset(dims_, SVS_FWD(indices)...);
    }

    ///
    /// @brief Access the specified element.
    ///
    /// @param indices The indices to access. Must satisfy ``sizeof...(indices) == ndims()``
    ///
    /// It is the callers responsibility to ensure that all indices are inbounds.
    ///
    template <typename... Is>
    reference at(Is&&... indices)
        requires(!is_const)
    {
        return *(data() + offset(SVS_FWD(indices)...));
    }

    /// @copydoc at()
    template <typename... Is> const_reference at(Is&&... indices) const {
        return *(data() + offset(SVS_FWD(indices)...));
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
    template <typename... Is>
    [[nodiscard]] span slice(Is&&... indices)
        requires(!is_const)
    {
        size_t o = offset(SVS_FWD(indices)..., 0);
        return span{data() + o, getsize<ndims() - 1>()};
    }

    /// @copydoc slice()
    template <typename... Is> [[nodiscard]] const_span slice(Is&&... indices) const {
        size_t o = offset(SVS_FWD(indices)..., 0);
        return const_span{data() + o, getsize<ndims() - 1>()};
    }

    [[nodiscard]] pointer data()
        requires(!is_const)
    {
        return pointer_;
    }
    [[nodiscard]] const_pointer data() const { return pointer_; }

    // Special Members
    constexpr DenseArray() = default;

    /////
    ///// Constructors
    /////

    explicit DenseArray(Dims dims, pointer ptr)
        : pointer_{ptr}
        , dims_{std::move(dims)} {}

    explicit DenseArray(Dims dims, View<T> view)
        : DenseArray{std::move(dims), view.ptr} {}

    // Iterator
    pointer begin()
        requires(!is_const)
    {
        return data();
    }
    const_pointer begin() const { return data(); }
    pointer end()
        requires(!is_const)
    {
        return data() + size();
    }
    const_pointer end() const { return data() + size(); }

    DenseArray<T, Dims, T*> view()
        requires(!std::is_const_v<T>)
    {
        return DenseArray<T, Dims, T*>(begin(), dims_);
    }

    DenseArray<const T, Dims, const T*> cview() const {
        return DenseArray<const T, Dims, const T*>(begin(), dims_);
    }
    DenseArray<const T, Dims, const T*> view() const { return cview(); }

  private:
  private:
    pointer pointer_{nullptr};
    [[no_unique_address]] Dims dims_{};
};

template <size_t I, typename T, typename Dims, typename Alloc>
size_t getsize(const DenseArray<T, Dims, Alloc>& array) {
    return array.template getsize<I>();
}

template <size_t I, typename T, typename Dims, typename Alloc>
constexpr size_t getextent(const DenseArray<T, Dims, Alloc>& /*unused*/) {
    return DenseArray<T, Dims, Alloc>::template getextent<I>();
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
/// Each argument must be either convertible to ``size_t`` or a ``svs::lib::Val`.
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
    return DenseArray<T, std::tuple<canonical_form_t<Dims>...>, lib::Allocator<T>>(
        std::tuple<canonical_form_t<Dims>...>(dims...)
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
template <typename T, typename Alloc, detail::IsDim... Dims>
    requires (!detail::IsDim<Alloc>)
auto make_dense_array(const Alloc& allocator, Dims... dims) {
    return DenseArray<T, std::tuple<canonical_form_t<Dims>...>, Alloc>(
        std::tuple<canonical_form_t<Dims>...>(dims...), allocator
    );
}

/////
///// Type Aliases
/////

/// @brief Dynamically sized, unresizeable vector.
template <typename T> using Vector = DenseArray<T, std::tuple<size_t>>;

/// @brief Dynamically sized matrix.
template <typename T> using Matrix = DenseArray<T, std::tuple<size_t, size_t>>;

/// @brief Dynamically sized, unresizeable vector.
template <typename T> using VectorView = DenseArray<T, std::tuple<size_t>, View<T>>;

/// @brief Dynamically sized matrix view.
template <typename T> using MatrixView = DenseArray<T, std::tuple<size_t, size_t>, View<T>>;

/// @brief Dynamically sized constant matrix view.
template <typename T>
using ConstMatrixView = DenseArray<const T, std::tuple<size_t, size_t>, View<const T>>;

} // namespace svs
