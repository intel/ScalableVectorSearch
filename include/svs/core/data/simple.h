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

// svs
#include "svs/concepts/data.h"
#include "svs/core/allocator.h"
#include "svs/core/data/abstract_io.h"
#include "svs/core/polymorphic_pointer.h"

#include "svs/lib/array.h"
#include "svs/lib/boundscheck.h"
#include "svs/lib/datatype.h"
#include "svs/lib/memory.h"
#include "svs/lib/prefetch.h"
#include "svs/lib/saveload.h"
#include "svs/lib/threads.h"
#include "svs/lib/uuid.h"

// stdlib
#include <span>
#include <type_traits>

namespace svs {
namespace data {

template <size_t M, size_t N> bool check_dims(size_t m, size_t n) {
    if constexpr (M == Dynamic || N == Dynamic) {
        return m == n;
    } else {
        static_assert(N == M);
        return true;
    }
}

/////
///// Simple Data
/////

// Generic save routine meant to be shared by the SimpleData specializations and the
// BlockedData.
//
// Ensures the two stay in-sync for the common parts.
template <data::ImmutableMemoryDataset Data> class GenericSaver {
  public:
    GenericSaver(const Data& data)
        : data_{data} {}

    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
    lib::SaveType save(const lib::SaveContext& ctx) const {
        using T = typename Data::element_type;
        // UUID used to identify the file.
        auto uuid = lib::UUID{};
        auto filename = ctx.generate_name("data");
        io::save(data_, io::NativeFile(filename), uuid);
        return lib::SaveType(
            toml::table({
                {"name", "uncompressed"},
                {"binary_file", filename.filename().c_str()},
                {"dims", prepare(data_.dimensions())},
                {"num_vectors", prepare(data_.size())},
                {"uuid", uuid.str()},
                {"eltype", name<datatype_v<T>>()},
            }),
            save_version
        );
    }

  private:
    const Data& data_;
};

// Forward Declaration
template <typename T, size_t Extent> class SimpleDataView;
template <typename T, size_t Extent> class ConstSimpleDataView;

/// The following properties hold:
/// * Vectors are stored contiguously in memory.
/// * All vectors have the same length.
template <typename T, size_t Extent, typename Base> class SimpleDataBase {
  public:
    /// The static dimensionality of the underlying data.
    static constexpr size_t extent = Extent;
    static constexpr bool supports_saving = true;
    /// The various instantiations of ``SimpleDataBase`` are expected to have dense layouts.
    /// Therefore, they are directly memory map compatible from appropriate files.
    static constexpr bool is_memory_map_compatible = true;
    using dim_type = std::tuple<size_t, dim_type_t<Extent>>;
    using array_type = DenseArray<T, dim_type, Base>;

    /// The allocator type used for this instance.
    using allocator_type = lib::memory::allocator_type_t<Base>;
    /// The data type used to encode each dimension of the stored vectors.
    using element_type = T;
    /// The type used to return a mutable handle to stored vectors.
    using value_type = std::span<element_type, Extent>;
    /// The type used to return a constant handle to stored vectors.
    using const_value_type = std::span<const element_type, Extent>;

    ///// Access Compatibility
    template <AccessMode = DefaultAccess> using mode_value_type = value_type;
    template <AccessMode = DefaultAccess> using mode_const_value_type = const_value_type;

    template <typename Distance> static Distance adapt_distance(const Distance& distance) {
        return threads::shallow_copy(distance);
    }

    template <typename Distance> static Distance self_distance(const Distance& distance) {
        return threads::shallow_copy(distance);
    }

    /////
    ///// Constructors
    /////

    SimpleDataBase() = default;

    explicit SimpleDataBase(array_type data)
        : data_{std::move(data)} {}

    explicit SimpleDataBase(Base base, size_t n_elements, size_t n_dimensions)
        : data_{std::move(base), n_elements, meta::forward_extent<Extent>(n_dimensions)} {}

    explicit SimpleDataBase(size_t n_elements, size_t n_dimensions)
        requires lib::memory::may_trivially_construct<allocator_type>
        : data_{allocator_type{}, n_elements, meta::forward_extent<Extent>(n_dimensions)} {}

    ///// Data Interface

    /// Return the number of entries in the dataset.
    size_t size() const { return getsize<0>(data_); }
    /// Return the number of dimensions for each entry in the dataset.
    size_t dimensions() const { return getsize<1>(data_); }

    ///
    /// @brief Return a constant handle to vector stored as position ``i``.
    ///
    /// **Preconditions:**
    ///
    /// * ``0 <= i < size()``
    ///
    template <AccessMode Mode = DefaultAccess>
    const_value_type get_datum(size_t i, Mode SVS_UNUSED(mode) = {}) const {
        return data_.slice(i);
    }

    ///
    /// @brief Return a mutable handle to vector stored as position ``i``.
    ///
    /// **NOTE**: Mutating the returned value directly may have unintended consequences.
    /// Perform with care.
    ///
    /// **Preconditions:**
    ///
    /// * ``0 <= i < size()``
    ///
    template <AccessMode Mode = DefaultAccess>
    value_type get_datum(size_t i, Mode SVS_UNUSED(mode) = {}) {
        return data_.slice(i);
    }

    /// Prefetch the vector at position ``i`` into the L1 cache.
    template <AccessMode Mode = DefaultAccess>
    void prefetch(size_t i, Mode SVS_UNUSED(mode) = {}) const {
        lib::prefetch(get_datum(i));
    }

    ///
    /// @brief Overwrite the contents of the vector at position ``i``.
    ///
    /// @param i The index at which to store the new data.
    /// @param datum The new vector in R^n to store.
    ///
    /// If ``U`` is the same type as ``element_type``, then this operation is simply a
    /// memory copy. Otherwise, ``lib::narrow`` will be used to convert each element of
    /// ``datum`` which may error if the conversion is not exact.
    ///
    /// **Preconditions:**
    ///
    /// * ``datum.size() == dimensions()``
    /// * ``0 <= i < size()``
    ///
    template <typename U, size_t N, AccessMode Mode = DefaultAccess>
    void set_datum(size_t i, std::span<U, N> datum, Mode SVS_UNUSED(mode) = {}) {
        if (!check_dims<Extent, N>(dimensions(), datum.size())) {
            throw ANNEXCEPTION(
                "Trying to assign vector of size ",
                datum.size(),
                " to a dataset with dimensionality ",
                dimensions(),
                '!'
            );
        }

        // Store the results.
        // Unfortunately, GCC is not smart enough to emit a memmove when `T` and `U` are
        // the same by inlining and optimizing `lib::relaxed_narrow`.
        //
        // Use ``relaxed_narrow`` to allow `float` arguments to `Float16` datasets.
        if constexpr (std::is_same_v<T, std::remove_const_t<U>>) {
            std::copy(datum.begin(), datum.end(), get_datum(i).begin());
        } else {
            std::transform(
                datum.begin(),
                datum.end(),
                get_datum(i).begin(),
                [](const U& u) { return lib::relaxed_narrow<T>(u); }
            );
        }
    }

    template <typename U, AccessMode Mode = DefaultAccess>
    void set_datum(size_t i, const std::vector<U>& datum, Mode mode = {}) {
        set_datum(i, lib::as_span(datum), mode);
    }

    const array_type& get_array() const { return data_; }

    /// Return the base pointer to the data.
    const T* data() const { return data_.data(); }
    /// Return the base pointer to the data.
    T* data() { return data_.data(); }

    // Return an iterator over each index in the dataset.
    threads::UnitRange<size_t> eachindex() const {
        return threads::UnitRange<size_t>{0, size()};
    }

    /// @brief Return a ConstSimpleDataView over this data.
    ConstSimpleDataView<T, Extent> cview() const {
        return ConstSimpleDataView<T, Extent>{data(), size(), dimensions()};
    }

    /// @brief Return a ConstSimpleDataView over this data.
    ConstSimpleDataView<T, Extent> view() const {
        return ConstSimpleDataView<T, Extent>{data(), size(), dimensions()};
    }

    /// @brief Return a SimpleDataView over this data.
    SimpleDataView<T, Extent> view() {
        return SimpleDataView<T, Extent>{data(), size(), dimensions()};
    }

    const Base& getbase() const { return data_.getbase(); }
    Base& getbase_mutable() { return data_.getbase_mutable(); }

    const T& data_begin() const { return data_.first(); }
    const T& data_end() const { return data_.last(); }

    ///// IO
    lib::SaveType save(const lib::SaveContext& ctx) const {
        return GenericSaver(*this).save(ctx);
    }

    // --- !! DANGER !! ---
    array_type&& acquire() { return std::move(data_); }

  private:
    array_type data_;
};

///
/// @brief A dense collection of vectors in R^n using a smart pointer for memory storage.
///
/// @tparam T The data type used to encode each dimension of the stored vectors.
/// @tparam Extent The compile-time dimensionality of each vector. Defaults to ``Dynamic``
///     is this information is not known.
/// @tparam Base The smart pointer-like object used to manage the lifetime of the memory
///     for the vectors.
///
template <typename T, size_t Extent = Dynamic, typename Base = lib::DefaultStorage<T>>
class SimpleData : public SimpleDataBase<T, Extent, Base> {
  public:
    // type aliases
    using parent_type = SimpleDataBase<T, Extent, Base>;
    /// The allocator type used for this instance.
    using allocator_type = typename parent_type::allocator_type;
    using array_type = typename parent_type::array_type;

    // constructors
    SimpleData() = default;
    SimpleData(array_type data)
        : parent_type{std::move(data)} {}

    ///
    /// @brief Construct a new dataset.
    ///
    /// @param n_elements The number of vectors in the dataset.
    /// @param n_dimensions The number of dimensions for each vector in the dataset.
    ///
    /// Assumes that the allocator associated with this class's smart pointer is default
    /// constructible.
    ///
    SimpleData(size_t n_elements, size_t n_dimensions)
        : parent_type(n_elements, n_dimensions) {}

    ///
    /// @brief Construct a new dataset.
    ///
    /// @param allocator The allocator instance to use for allocation.
    /// @param n_elements The number of vectors in the dataset.
    /// @param n_dimensions The number of dimensions for each vector in the dataset.
    ///
    template <typename Allocator>
        requires std::is_same_v<Allocator, allocator_type>
    SimpleData(Allocator allocator, size_t n_elements, size_t n_dimensions)
        : parent_type{make_dense_array<T>(
              allocator, n_elements, meta::forward_extent<Extent>(n_dimensions)
          )} {}
};

// Deduction Guide
template <typename T, typename Dims, typename Base>
SimpleData(DenseArray<T, Dims, Base>)
    -> SimpleData<T, DenseArray<T, Dims, Base>::template getextent<1>(), Base>;

/////
///// Specializations
/////

///
/// @brief A non-owning mutable view of a dense collection of vectors in R^n.
///
/// @tparam T The data type used to encode each dimension of the stored vectors.
/// @tparam Extent The compile-time dimensionality of each vector. Defaults to ``Dynamic``
///     is this information is not known.
///
template <typename T, size_t Extent = Dynamic>
class SimpleDataView : public SimpleDataBase<T, Extent, T*> {
  public:
    // type aliases
    using parent_type = SimpleDataBase<T, Extent, T*>;

    ///
    /// @brief Construct a non-owning dense dataset view beginning at ``base``.
    ///
    /// @param base The base pointer for the data to construct a view over.
    /// @param n_elements The number of vectors in the dataset.
    /// @param n_dimensions The number of dimensions in each vector.
    ///
    /// If ``Extent != svs::Dynamic``, then ``n_dimensions == Extent`` must hold.
    /// Otherwise, the behavior is undefined.
    ///
    SimpleDataView(T* base, size_t n_elements, size_t n_dimensions)
        : parent_type{base, n_elements, n_dimensions} {}

    /// @brief Construct a non-owning mutable view over the provided dataset.
    template <typename Base>
    SimpleDataView(SimpleData<T, Extent, Base>& other)
        : SimpleDataView(other.data(), other.size(), other.dimensions()) {}
};

// deduction guide
template <typename T, size_t Extent, typename Base>
SimpleDataView(SimpleDataBase<T, Extent, Base>&) -> SimpleDataView<T, Extent>;

///
/// @brief A non-owning constant view of a dense collection of vectors in R^n.
///
/// @tparam T The data type used to encode each dimension of the stored vectors.
/// @tparam Extent The compile-time dimensionality of each vector. Defaults to ``Dynamic``
///     is this information is not known.
///
template <typename T, size_t Extent = Dynamic>
class ConstSimpleDataView : public SimpleDataBase<const T, Extent, const T*> {
  public:
    // type aliases
    using parent_type = SimpleDataBase<const T, Extent, const T*>;

    // constructors
    ConstSimpleDataView() = default;

    ///
    /// @brief Construct a non-owning constant dense dataset view beginning at ``base``.
    ///
    /// @param base The base pointer for the data to construct a view over.
    /// @param n_elements The number of vectors in the dataset.
    /// @param n_dimensions The number of dimensions in each vector.
    ///
    /// If ``Extent != svs::Dynamic``, then ``n_dimensions == Extent`` must hold.
    /// Otherwise, the behavior is undefined.
    ///
    ConstSimpleDataView(const T* base, size_t n_elements, size_t n_dimensions)
        : parent_type{base, n_elements, n_dimensions} {}

    /// @brief Construct a non-owning mutable view over the provided dataset.
    template <typename Base>
    ConstSimpleDataView(const SimpleData<T, Extent, Base>& other)
        : ConstSimpleDataView(other.data(), other.size(), other.dimensions()) {}
};

// deduction guide
template <typename T, size_t Extent, typename Base>
ConstSimpleDataView(SimpleData<T, Extent, Base>&) -> ConstSimpleDataView<T, Extent>;

///
/// @brief A dense collection of vectors in R^n using a polymorphic allocator for storage.
///
/// @tparam T The data type used to encode each dimension of the stored vectors.
/// @tparam Extent The compile-time dimensionality of each vector. Defaults to ``Dynamic``
///     is this information is not known.
///
/// In general, this class should be used if polymorphism of the allocator is desired but
/// not necessarily relevant for performance.
///
template <typename T, size_t Extent = Dynamic>
class SimplePolymorphicData : public SimpleDataBase<T, Extent, PolymorphicPointer<T>> {
  public:
    // type aliases
    using parent_type = SimpleDataBase<T, Extent, PolymorphicPointer<T>>;
    using dim_type = typename parent_type::dim_type;
    using array_type = typename parent_type::array_type;

    // constructors
    explicit SimplePolymorphicData(array_type data)
        : parent_type{std::move(data)} {}

    template <typename OtherBase>
    explicit SimplePolymorphicData(DenseArray<T, dim_type, OtherBase> data)
        : parent_type{polymorph(std::move(data))} {}

    ///
    /// @brief Convert the dataset to a SimplePolymorphicData
    ///
    /// @param other The other dataset to acquire.
    ///
    /// Takes ownership of the other dataset, using type-erasure on its storage pointer.
    ///
    template <typename OtherBase>
    explicit SimplePolymorphicData(SimpleData<T, Extent, OtherBase> other)
        : parent_type{polymorph(other.acquire())} {}

    ///
    /// @brief Construct a dataset using the provided allocator.
    ///
    /// @param allocator The allocator to use for memory allocation.
    /// @param n_elements The number of element to be in the final dataset.
    /// @param n_dimensions The dimensionality of each vector in the dataset.
    ///
    template <lib::memory::MemoryAllocator Allocator>
    explicit SimplePolymorphicData(
        Allocator allocator, size_t n_elements, size_t n_dimensions
    )
        : parent_type{polymorph(make_dense_array<T>(
              std::move(allocator), n_elements, meta::forward_extent<Extent>(n_dimensions)
          ))} {}

    ///
    /// @brief Construct a dataset using an unspecified default allocator.
    ///
    /// @param n_elements The number of element to be in the final dataset.
    /// @param n_dimensions The dimensionality of each vector in the dataset.
    ///
    explicit SimplePolymorphicData(size_t n_elements, size_t n_dimensions)
        : SimplePolymorphicData(lib::DefaultAllocator{}, n_elements, n_dimensions) {}
};

template <typename T1, size_t E1, typename T2, size_t E2>
bool operator==(
    const SimplePolymorphicData<T1, E1>& x, const SimplePolymorphicData<T2, E2>& y
) {
    if ((x.size() != y.size()) || (x.dimensions() != y.dimensions())) {
        return false;
    }

    for (size_t i = 0, imax = x.size(); i < imax; ++i) {
        const auto& xdata = x.get_datum(i);
        const auto& ydata = y.get_datum(i);
        if (!std::equal(xdata.begin(), xdata.end(), ydata.begin())) {
            return false;
        }
    }
    return true;
}

// deduction guides
template <typename T, typename Dims, typename Base>
SimplePolymorphicData(DenseArray<T, Dims, Base> data)
    -> SimplePolymorphicData<T, DenseArray<T, Dims, Base>::template getextent<1>()>;

template <typename T, size_t Extent, typename Base>
SimplePolymorphicData(SimpleData<T, Extent, Base> data) -> SimplePolymorphicData<T, Extent>;

/////
///// Builders
/////

namespace detail {
template <typename T> inline constexpr bool is_variant = false;
template <typename... Ts> inline constexpr bool is_variant<std::variant<Ts...>> = true;
} // namespace detail

// Instantiators.
template <typename Allocator = lib::DefaultAllocator> class PolymorphicBuilder {
  public:
    PolymorphicBuilder() = default;
    PolymorphicBuilder(const Allocator& allocator)
        : allocator_{allocator} {}

    template <typename T, size_t Extent = Dynamic>
    using return_type = data::SimplePolymorphicData<T, Extent>;

    // Allocate a ``data::SimplePolymorphicData`` of an appropriate size.
    template <typename T, size_t Extent = Dynamic>
    return_type<T, Extent> build(size_t size, size_t dimensions) const {
        if constexpr (detail::is_variant<Allocator>) {
            return std::visit(
                [&](auto alloc) {
                    return data::SimplePolymorphicData<T, Extent>(alloc, size, dimensions);
                },
                allocator_
            );
        } else {
            return data::SimplePolymorphicData<T, Extent>(allocator_, size, dimensions);
        }
    }

    // Pre-processing hook during reloading.
    // Nothing to do for the PolymorphicBuilder.
    void load_hook(const toml::table&) const {}

  private:
    Allocator allocator_{};
};

template <typename Builder, typename T, size_t Extent>
using builder_return_type = typename Builder::template return_type<T, Extent>;

template <typename T, size_t Extent, typename Builder>
builder_return_type<Builder, T, Extent>
build(const Builder& builder, size_t size, size_t dimensions) {
    return builder.template build<T, Extent>(size, dimensions);
}

} // namespace data
} // namespace svs
