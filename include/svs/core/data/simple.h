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
    return std::filesystem::is_directory(path) || maybe_config_file(path);
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

    template <data::ImmutableMemoryDataset Data>
    static lib::SaveTable save(const Data& data, const lib::SaveContext& ctx) {
        using T = typename Data::element_type;
        // UUID used to identify the file.
        auto uuid = lib::UUID{};
        auto filename = ctx.generate_name("data");
        io::save(data, io::NativeFile(filename), uuid);
        return lib::SaveTable(
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
    static lib::lazy_result_t<F, size_t, size_t> load(
        const toml::table& table,
        const lib::LoadContext& ctx,
        const lib::Version& version,
        const F& lazy
    ) {
        if (version != save_version) {
            throw ANNException("Version mismatch!");
        }

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
        auto binaryfile = io::find_uuid(ctx.get_directory(), uuid);
        if (!binaryfile.has_value()) {
            throw ANNEXCEPTION("Could not open file with uuid {}!", uuid.str());
        }
        return io::load_dataset(binaryfile.value(), lazy);
    }
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
  public:
    /// The static dimensionality of the underlying data.
    static constexpr size_t extent = Extent;

    /// The various instantiations of ``SimpleDataBase`` are expected to have dense layouts.
    /// Therefore, they are directly memory map compatible from appropriate files.
    static constexpr bool is_memory_map_compatible = true;
    static constexpr bool is_view = is_view_type_v<Alloc>;
    static constexpr bool is_const = std::is_const_v<T>;

    using dim_type = std::tuple<size_t, dim_type_t<Extent>>;
    using array_type = DenseArray<T, dim_type, Alloc>;

    // /// The allocator type used for this instance.
    // using allocator_type = lib::memory::allocator_type_t<Base>;
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
        : data_{std::move(data)} {}

    explicit SimpleData(size_t n_elements, size_t n_dimensions, const Alloc& allocator)
        : data_{
              make_dims(n_elements, meta::forward_extent<Extent>(n_dimensions)),
              allocator} {}

    explicit SimpleData(size_t n_elements, size_t n_dimensions)
        : SimpleData(n_elements, n_dimensions, Alloc()) {}

    // View compatibility layers.
    explicit SimpleData(T* ptr, size_t n_elements, size_t n_dimensions)
        requires(is_view)
        : SimpleData(n_elements, n_dimensions, View{ptr}) {}

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

    ///
    /// @brief Reload a previously saved dataset.
    ///
    /// @param table The table containing saved hyper parameters.
    /// @param ctx The current load context.
    /// @param version The version number used when saving the dataset.
    /// @param allocator Allocator instance to use upon reloading.
    ///
    /// This method is implicitly called when using
    /// @code{cpp}
    /// svs::lib::load_from_disk<svs::data::SimpleData<T, Extent>>("directory");
    /// @endcode
    ///
    static SimpleData load(
        const toml::table& table,
        const lib::LoadContext& ctx,
        const lib::Version& version,
        const allocator_type& allocator = {}
    )
        requires(!is_view)
    {
        return GenericSerializer::load<T>(
            table, ctx, version, lib::Lazy([&](size_t n_elements, size_t n_dimensions) {
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
        } else {
            return io::auto_load<T>(
                path, lib::Lazy([&](size_t n_elements, size_t n_dimensions) {
                    return SimpleData(n_elements, n_dimensions, allocator);
                })
            );
        }
    }

  private:
    array_type data_;
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
            make_dims(blocksize().value(), meta::forward_extent<Extent>(dimensions())),
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
    template <typename I, typename A, threads::ThreadPool Pool>
    void compact(
        const std::vector<I, A>& new_to_old, Pool& threadpool, size_t batchsize = 1'000'000
    ) {
        // Alllocate scratch space.
        batchsize = std::min(batchsize, size());
        auto buffer = data::SimpleData<T, Extent>(batchsize, dimensions());
        compact_data(*this, buffer, new_to_old, threadpool);
    }

    template <typename I, typename A>
    void compact(const std::vector<I, A>& new_to_old, size_t batchsize = 1'000'000) {
        auto pool = threads::SequentialThreadPool();
        compact(new_to_old, pool, batchsize);
    }

    ///// Saving
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        return GenericSerializer::save(*this, ctx);
    }

    static SimpleData load(
        const toml::table& table,
        const lib::LoadContext& ctx,
        const lib::Version& version,
        const Blocked<Alloc>& allocator = {}
    ) {
        return GenericSerializer::load<T>(
            table,
            ctx,
            version,
            lib::Lazy([&allocator](size_t n_elements, size_t n_dimensions) {
                return SimpleData(n_elements, n_dimensions, allocator);
            })
        );
    }

    static SimpleData
    load(const std::filesystem::path& path, const Blocked<Alloc>& allocator = {}) {
        if (detail::is_likely_reload(path)) {
            return lib::load_from_disk<SimpleData>(path, allocator);
        } else {
            return io::auto_load<T>(
                path, lib::Lazy([&allocator](size_t n_elements, size_t n_dimensions) {
                    return SimpleData(n_elements, n_dimensions, allocator);
                })
            );
        }
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
