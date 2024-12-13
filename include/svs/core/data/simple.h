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

// svs
#include "svs/concepts/data.h"
#include "svs/core/allocator.h"
#include "svs/core/compact.h"
#include "svs/core/data/io.h"

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

namespace detail {
inline bool is_likely_reload(const std::filesystem::path& path) {
    return std::filesystem::is_directory(path) || config_file_by_extension(path);
}
} // namespace detail

/////
///// Simple Data
/////

// Generic save routine meant to be shared by the SimpleData specializations and the
// BlockedData.
//
// Ensures the two stay in-sync for the common parts.
class GenericSerializer {
  public:
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
    static constexpr std::string_view serialization_schema = "uncompressed_data";

    static constexpr bool
    check_compatibility(std::string_view schema, lib::Version version) {
        return schema == serialization_schema && version == save_version;
    }

    template <data::ImmutableMemoryDataset Data>
    static lib::SaveTable save(const Data& data, const lib::SaveContext& ctx) {
        using T = typename Data::element_type;
        // UUID used to identify the file.
        auto uuid = lib::UUID{};
        auto filename = ctx.generate_name("data");
        io::save(data, io::NativeFile(filename), uuid);
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {
                {"name", "uncompressed"},
                {"binary_file", lib::save(filename.filename())},
                {"dims", lib::save(data.dimensions())},
                {"num_vectors", lib::save(data.size())},
                {"uuid", uuid.str()},
                {"eltype", lib::save(datatype_v<T>)},
            }
        );
    }

    template <typename T, lib::LazyInvocable<size_t, size_t> F>
    static lib::lazy_result_t<F, size_t, size_t>
    load(const lib::LoadTable& table, const F& lazy) {
        auto datatype = lib::load_at<DataType>(table, "eltype");
        if (datatype != datatype_v<T>) {
            throw ANNEXCEPTION(
                "Trying to load an uncompressed dataset with element types {} to a dataset "
                "with element types {}.",
                name(datatype),
                name<datatype_v<T>>()
            );
        }

        // Now that this is out of the way, resolve the file and load the data.
        auto uuid = lib::load_at<lib::UUID>(table, "uuid");
        auto binaryfile = io::find_uuid(table.context().get_directory(), uuid);
        if (!binaryfile.has_value()) {
            throw ANNEXCEPTION("Could not open file with uuid {}!", uuid.str());
        }
        return io::load_dataset(binaryfile.value(), lazy);
    }
};

struct Matcher {
    // Compatibility check is routed through the GenericSerializer.
    static bool check_load_compatibility(std::string_view schema, lib::Version version) {
        return GenericSerializer::check_compatibility(schema, version);
    }

    // Support direct loading for the common filetypes.
    static bool can_load_direct(
        const std::filesystem::path& path,
        svs::DataType SVS_UNUSED(type_hint) = svs::DataType::undef,
        size_t SVS_UNUSED(dims_hint) = Dynamic
    ) {
        return svs::io::special_by_file_extension(std::string_view(path.native()));
    }

    ///// try_load and friends
    static Matcher load_direct(
        const std::filesystem::path& path,
        svs::DataType type_hint = svs::DataType::undef,
        size_t dims_hint = Dynamic
    ) {
        if (type_hint == svs::DataType::undef) {
            throw ANNEXCEPTION("Cannot deduce the element type of raw file {}.", path);
        }

        size_t dims = io::deduce_dimensions(path);
        if (dims_hint != Dynamic && dims != dims_hint) {
            throw ANNEXCEPTION(
                "Dims hint {} does not match deduced dimensions of {}!", dims_hint, dims
            );
        }
        return Matcher{.eltype = type_hint, .dims = dims};
    }

    // Don't support deduction on the try-load path.
    // Deduction throws too many exceptions to handle corectly right now.
    static lib::TryLoadResult<Matcher> try_load_direct(
        const std::filesystem::path& SVS_UNUSED(path),
        svs::DataType SVS_UNUSED(type_hint) = svs::DataType::undef,
        size_t SVS_UNUSED(dims_hint) = Dynamic
    ) {
        return lib::Unexpected(lib::TryLoadFailureReason::Other);
    }

    ///// load and friends
    static Matcher load(
        const lib::ContextFreeLoadTable& table,
        svs::DataType type_hint = svs::DataType::undef,
        size_t dims_hint = Dynamic
    ) {
        auto matcher = Matcher{
            .eltype = SVS_LOAD_MEMBER_AT(table, eltype),
            .dims = SVS_LOAD_MEMBER_AT(table, dims)};

        // Perform a sanity check on the arguments.
        if (type_hint != DataType::undef && type_hint != matcher.eltype) {
            throw ANNEXCEPTION(
                "A dataset type hint of {} was given but the discovered type is {}!",
                type_hint,
                matcher.eltype
            );
        }

        if (dims_hint != Dynamic && dims_hint != matcher.dims) {
            throw ANNEXCEPTION(
                "Dataset dimensionality hint {} does not match discovered value {}!",
                dims_hint,
                matcher.dims
            );
        }
        return matcher;
    }

    /// @brief Try to load a ``Check`` from a ``table``.
    static lib::TryLoadResult<Matcher> try_load(const lib::ContextFreeLoadTable& table) {
        // Compatibility check performed implicitly by saving/loading infrastructure.
        return load(table);
    }

  public:
    /// The type of each element of each vector.
    DataType eltype;
    /// The number of dimensions in each vector.
    size_t dims;
};

// Forward Declaration
template <typename T, size_t Extent, typename Alloc> class SimpleData;

template <typename T, size_t Extent = Dynamic>
using SimpleDataView = SimpleData<T, Extent, View<T>>;

template <typename T, size_t Extent = Dynamic>
using ConstSimpleDataView = SimpleData<const T, Extent, View<const T>>;

/// The following properties hold:
/// * Vectors are stored contiguously in memory.
/// * All vectors have the same length.
template <typename T, size_t Extent = Dynamic, typename Alloc = lib::Allocator<T>>
class SimpleData {
    static_assert(std::is_trivial_v<T>, "SimpleData may only contain trivial types!");

  public:
    /// The static dimensionality of the underlying data.
    static constexpr size_t extent = Extent;

    /// The various instantiations of ``SimpleData`` are expected to have dense layouts.
    /// Therefore, they are directly memory map compatible from appropriate files.
    ///
    /// However, some specializations (such as the blocked dataset) are not necessarily
    /// memory map compatible.
    static constexpr bool is_memory_map_compatible = true;

    /// Return whether or not this is a non-owning view of the underlying data.
    static constexpr bool is_view = is_view_type_v<Alloc>;
    /// Return whether or not this class is allowed to mutate its backing data.
    static constexpr bool is_const = std::is_const_v<T>;

    using dim_type = std::tuple<size_t, dim_type_t<Extent>>;
    using array_type = DenseArray<T, dim_type, Alloc>;

    /// The allocator type used for this instance.
    using allocator_type = Alloc;
    /// The data type used to encode each dimension of the stored vectors.
    using element_type = T;
    /// The type used to return a mutable handle to stored vectors.
    using value_type = std::span<element_type, Extent>;
    /// The type used to return a constant handle to stored vectors.
    using const_value_type = std::span<const element_type, Extent>;

    /// Return the underlying allocator.
    const allocator_type& get_allocator() const { return data_.get_allocator(); }

    /////
    ///// Constructors
    /////

    SimpleData() = default;

    explicit SimpleData(array_type data)
        : data_{std::move(data)}
        , size_{getsize<0>(data_)} {}

    explicit SimpleData(size_t n_elements, size_t n_dimensions, const Alloc& allocator)
        : data_{make_dims(n_elements, lib::forward_extent<Extent>(n_dimensions)), allocator}
        , size_{n_elements} {}

    explicit SimpleData(size_t n_elements, size_t n_dimensions)
        : SimpleData(n_elements, n_dimensions, Alloc()) {}

    // View compatibility layers.
    explicit SimpleData(T* ptr, size_t n_elements, size_t n_dimensions)
        requires(is_view)
        : SimpleData(n_elements, n_dimensions, View{ptr}) {}

    /// Construct a view over the array using a checked cast.
    explicit SimpleData(AnonymousArray<2> array)
        requires(is_view && is_const)
        : SimpleData(array.size(0), array.size(1), View{get<T>(array)}) {}

    ///// Conversions
    explicit operator AnonymousArray<2>() const {
        return AnonymousArray<2>(data(), size(), dimensions());
    }

    ///// Data Interface

    /// Return the number of entries in the dataset.
    size_t size() const { return size_; }
    /// Return the maximum number of entries this dataset can hold.
    size_t capacity() const { return getsize<0>(data_); }
    /// Return the number of dimensions for each entry in the dataset.
    size_t dimensions() const { return getsize<1>(data_); }

    ///
    /// @brief Return a constant handle to vector stored as position ``i``.
    ///
    /// **Preconditions:**
    ///
    /// * ``0 <= i < size()``
    ///
    const_value_type get_datum(size_t i) const { return data_.slice(i); }

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
    value_type get_datum(size_t i) { return data_.slice(i); }

    /// Prefetch the vector at position ``i`` into the L1 cache.
    void prefetch(size_t i) const { lib::prefetch(get_datum(i)); }

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
    template <typename U, size_t N> void set_datum(size_t i, std::span<U, N> datum) {
        if (!check_dims<Extent, N>(dimensions(), datum.size())) {
            throw ANNEXCEPTION(
                "Trying to assign vector of size {} to a dataset with dimensionality {}.",
                datum.size(),
                dimensions()
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

    template <typename U, typename A>
    void set_datum(size_t i, const std::vector<U, A>& datum) {
        set_datum(i, lib::as_span(datum));
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
        return ConstSimpleDataView<T, Extent>{size(), dimensions(), View{data()}};
    }

    /// @brief Return a ConstSimpleDataView over this data.
    ConstSimpleDataView<T, Extent> view() const {
        return ConstSimpleDataView<T, Extent>{size(), dimensions(), View{data()}};
    }

    /// @brief Return a SimpleDataView over this data.
    SimpleDataView<T, Extent> view() {
        return SimpleDataView<T, Extent>{size(), dimensions(), View{data()}};
    }

    const T& data_begin() const { return data_.first(); }
    const T& data_end() const { return data_.last(); }

    ///// IO
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        return GenericSerializer::save(*this, ctx);
    }

    static bool check_load_compatibility(std::string_view schema, lib::Version version) {
        return GenericSerializer::check_compatibility(schema, version);
    }

    ///
    /// @brief Reload a previously saved dataset.
    ///
    /// @param table The table containing saved hyper parameters.
    /// @param allocator Allocator instance to use upon reloading.
    ///
    /// This method is implicitly called when using
    /// @code{cpp}
    /// svs::lib::load_from_disk<svs::data::SimpleData<T, Extent>>("directory");
    /// @endcode
    ///
    static SimpleData
    load(const lib::LoadTable& table, const allocator_type& allocator = {})
        requires(!is_view)
    {
        return GenericSerializer::load<T>(
            table, lib::Lazy([&](size_t n_elements, size_t n_dimensions) {
                return SimpleData(n_elements, n_dimensions, allocator);
            })
        );
    }

    ///
    /// @brief Try to automatically load the dataset.
    ///
    /// @param path The filepath to a dataset on disk.
    /// @param allocator The allocator instance to use when constructing this class.
    ///
    /// The argument ``path`` can point to:
    /// * The directory previously used to save a dataset (or the config file of such a
    ///   directory).
    /// * A ".[f/b/i]vecs" file.
    ///
    static SimpleData
    load(const std::filesystem::path& path, const allocator_type& allocator = {})
        requires(!is_view)
    {
        if (detail::is_likely_reload(path)) {
            return lib::load_from_disk<SimpleData>(path, allocator);
        }
        // Try loading directly.
        return io::auto_load<T>(
            path, lib::Lazy([&](size_t n_elements, size_t n_dimensions) {
                return SimpleData(n_elements, n_dimensions, allocator);
            })
        );
    }

    ///
    /// @brief Resize the dataset to the new size.
    ///
    /// Causes a reallocation if ``new_size > capacity()``.
    /// Growing and shrinking are performed at the end the valid range.
    ///
    /// **NOTE**: Resizing that triggers a reallocation will invalidate *all* previously
    /// obtained pointers!.
    ///
    void resize(size_t new_size)
        requires(!is_view)
    {
        resize_impl(new_size, false);
    }

    ///
    /// @brief Requests the removal of unused capacity.
    ///
    /// It is a non-binding request to reduce ``capacity()`` to ``size()``.
    /// If relocation occurs, all iterators and previously obtained datums are invalidated.
    ///
    void shrink_to_fit()
        requires(!is_view)
    {
        resize_impl(size(), true);
    }

    template <std::integral I, threads::ThreadPool Pool>
        requires(!is_const)
    void compact(
        std::span<const I> new_to_old, Pool& threadpool, size_t batchsize = 1'000'000
    ) {
        // Alllocate scratch space.
        batchsize = std::min(batchsize, size());
        auto buffer = data::SimpleData<T, Extent>(batchsize, dimensions());
        compact_data(*this, buffer, new_to_old, threadpool);
    }

    template <std::integral I>
        requires(!is_const)
    void compact(std::span<const I> new_to_old, size_t batchsize = 1'000'000) {
        auto pool = threads::SequentialThreadPool();
        compact(new_to_old, pool, batchsize);
    }

  private:
    void resize_impl(size_t new_size, bool force_reallocate) {
        bool forced = force_reallocate && (capacity() != size());
        if (forced || new_size > capacity()) {
            auto new_data = array_type{
                svs::make_dims(new_size, lib::forward_extent<Extent>(dimensions())),
                get_allocator()};

            // Copy our contents into the new array.
            // Since the backing array is dense, we can use `memcpy`.
            std::memcpy(new_data.data(), data(), sizeof(T) * size() * dimensions());

            // Swap out the internal buffer.
            data_ = std::move(new_data);
        }
        // Any change to the underlying buffer has been performed.
        // We are now safe to change size.
        size_ = new_size;
    }

    ///// Members
    array_type data_;
    size_t size_;
};

template <typename T1, size_t E1, typename A1, typename T2, size_t E2, typename A2>
bool operator==(const SimpleData<T1, E1, A1>& x, const SimpleData<T2, E2, A2>& y) {
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

/////
///// Specialization for Blocked.
/////

struct BlockingParameters {
  public:
    static constexpr lib::PowerOfTwo default_blocksize_bytes{30};

    friend bool operator==(const BlockingParameters&, const BlockingParameters&) = default;

  public:
    lib::PowerOfTwo blocksize_bytes = default_blocksize_bytes;
};

template <typename Alloc> class Blocked {
  public:
    using allocator_type = Alloc;
    const allocator_type& get_allocator() const { return allocator_; }
    const BlockingParameters& parameters() const { return parameters_; }

    constexpr Blocked() = default;
    explicit Blocked(const allocator_type& alloc)
        : allocator_{alloc} {}
    explicit Blocked(const BlockingParameters& parameters)
        : parameters_{parameters} {}
    explicit Blocked(const BlockingParameters& parameters, const allocator_type& alloc)
        : parameters_{parameters}
        , allocator_{alloc} {}

    // Enable rebinding of allocators.
    template <typename U> friend class Blocked;
    template <typename U>
    Blocked(const Blocked<U>& other)
        : parameters_{other.parameters_} {}

  private:
    BlockingParameters parameters_{};
    Alloc allocator_{};
};

///
/// @brief A specialization of ``SimpleData`` for large-scale dynamic datasets.
///
template <typename T, size_t Extent, typename Alloc>
class SimpleData<T, Extent, Blocked<Alloc>> {
  public:
    ///// Static Members

    ///
    /// Default block size in bytes.
    ///
    static constexpr bool supports_saving = true;

    // Type Aliases
    using dim_type = std::tuple<size_t, dim_type_t<Extent>>;
    using allocator_type = Blocked<Alloc>;
    using inner_allocator_type = Alloc;
    using array_type = DenseArray<T, dim_type, inner_allocator_type>;

    /// Return the underlying allocator.
    const allocator_type& get_allocator() const { return allocator_; }

    // value types
    using element_type = T;
    using value_type = std::span<T, Extent>;
    using const_value_type = std::span<const T, Extent>;

    ///// Constructors
    SimpleData(size_t n_elements, size_t n_dimensions, const Blocked<Alloc>& alloc)
        : blocksize_{lib::prevpow2(
              alloc.parameters().blocksize_bytes.value() / (sizeof(T) * n_dimensions)
          )}
        , blocks_{}
        , dimensions_{n_dimensions}
        , size_{n_elements}
        , allocator_{alloc} {
        size_t elements_per_block = blocksize_.value();
        size_t num_blocks = lib::div_round_up(n_elements, elements_per_block);
        blocks_.reserve(num_blocks);
        for (size_t i = 0; i < num_blocks; ++i) {
            add_block();
        }
    }

    SimpleData(size_t n_elements, size_t n_dimensions)
        : SimpleData{n_elements, n_dimensions, Blocked<Alloc>()} {}

    ///
    /// Convert a linear index into an inner-outer index to access the blocked dataset.
    /// Returns a pair `p` where:
    /// - `p.first` is the block index.
    /// - `p.second` is the index within the block.
    ///
    std::pair<size_t, size_t> resolve(size_t i) const {
        return std::pair<size_t, size_t>{i / blocksize_, i % blocksize_};
    }

    ///
    /// Return the blocksize with reference to the stored data vectors.
    ///
    lib::PowerOfTwo blocksize() const { return blocksize_; }

    ///
    /// Return the blocksize with respect to bytes.
    ///
    lib::PowerOfTwo blocksize_bytes() const {
        return allocator_.parameters().blocksize_bytes;
    }

    ///
    /// Return the number of blocks in the dataset.
    ///
    size_t num_blocks() const { return blocks_.size(); }

    ///
    /// Return the maximum number of data vectors that can be stored before a new block is
    /// required.
    ///
    size_t capacity() const { return num_blocks() * blocksize(); }

    ///
    /// Return an iterator over each index in the dataset.
    ///
    threads::UnitRange<size_t> eachindex() const {
        return threads::UnitRange<size_t>{0, size()};
    }

    ///
    /// Add a new data block to the end of the current collection of blocks.
    ///
    void add_block() {
        blocks_.emplace_back(
            make_dims(blocksize().value(), lib::forward_extent<Extent>(dimensions())),
            allocator_.get_allocator()
        );
    }

    ///
    /// Remove a data block from the end of the block list.
    ///
    void drop_block() {
        if (!blocks_.empty()) {
            blocks_.pop_back();
        }
    }

    ///
    /// Resizing
    ///
    void resize(size_t new_size) {
        if (new_size > size()) {
            // Add blocks until there is sufficient capacity.
            while (new_size > capacity()) {
                add_block();
            }
            size_ = new_size;
        } else if (new_size < size()) {
            // Reset size then drop blocks until the new size is within the last block.
            size_ = new_size;
            while (capacity() - blocksize().value() > new_size) {
                drop_block();
            }
        }
    }

    void shrink_to_fit() {
        // We already shrink when down-sizing, so ``shink_to_fit`` becomes a no-op.
    }

    /////
    ///// Dataset API
    /////

    size_t size() const { return size_; }
    constexpr size_t dimensions() const {
        if constexpr (Extent != Dynamic) {
            return Extent;
        } else {
            return dimensions_;
        }
    }

    const_value_type get_datum(size_t i) const {
        auto [block_id, data_id] = resolve(i);
        return getindex(blocks_, block_id).slice(data_id);
    }

    value_type get_datum(size_t i) {
        auto [block_id, data_id] = resolve(i);
        return getindex(blocks_, block_id).slice(data_id);
    }

    void prefetch(size_t i) const { lib::prefetch(get_datum(i)); }

    template <typename U, size_t OtherExtent>
    void set_datum(size_t i, std::span<U, OtherExtent> datum) {
        if constexpr (checkbounds_v) {
            if (datum.size() != dimensions()) {
                throw ANNEXCEPTION(
                    "Datum with dimensions {} is not equal to internal dimensions {}!",
                    datum.size(),
                    dimensions_
                );
            }
        }

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

    template <typename U, typename A> void set_datum(size_t i, const std::vector<U, A>& v) {
        set_datum(i, lib::as_span(v));
    }

    ///
    /// Construct an identical copy of the dataset.
    /// Not implemented as a copy constructor to avoid unintentional copies.
    ///
    SimpleData copy() const {
        SimpleData other{size(), dimensions(), allocator_};
        for (const auto& i : eachindex()) {
            other.set_datum(i, get_datum(i));
        }
        return other;
    }

    ///// Compaction
    template <std::integral I, threads::ThreadPool Pool>
    void
    compact(std::span<const I> new_to_old, Pool& threadpool, size_t batchsize = 1'000'000) {
        // Alllocate scratch space.
        batchsize = std::min(batchsize, size());
        auto buffer = data::SimpleData<T, Extent>(batchsize, dimensions());
        compact_data(*this, buffer, new_to_old, threadpool);
    }

    template <std::integral I>
    void compact(std::span<const I> new_to_old, size_t batchsize = 1'000'000) {
        auto pool = threads::SequentialThreadPool();
        compact(new_to_old, pool, batchsize);
    }

    ///// Saving
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        return GenericSerializer::save(*this, ctx);
    }

    static bool check_load_compatibility(std::string_view schema, lib::Version version) {
        return GenericSerializer::check_compatibility(schema, version);
    }

    static SimpleData
    load(const lib::LoadTable& table, const Blocked<Alloc>& allocator = {}) {
        return GenericSerializer::load<T>(
            table, lib::Lazy([&allocator](size_t n_elements, size_t n_dimensions) {
                return SimpleData(n_elements, n_dimensions, allocator);
            })
        );
    }

    static SimpleData
    load(const std::filesystem::path& path, const Blocked<Alloc>& allocator = {}) {
        if (detail::is_likely_reload(path)) {
            return lib::load_from_disk<SimpleData>(path, allocator);
        }
        // Try loading directly.
        return io::auto_load<T>(
            path, lib::Lazy([&allocator](size_t n_elements, size_t n_dimensions) {
                return SimpleData(n_elements, n_dimensions, allocator);
            })
        );
    }

  private:
    // The blocksize in terms of number of vectors.
    lib::PowerOfTwo blocksize_;
    std::vector<array_type> blocks_;
    size_t dimensions_;
    size_t size_;
    Blocked<Alloc> allocator_;
};

template <typename T, size_t Extent = Dynamic, typename Alloc = lib::Allocator<T>>
using BlockedData = SimpleData<T, Extent, Blocked<Alloc>>;

} // namespace data
} // namespace svs
