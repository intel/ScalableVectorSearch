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
#include "svs/core/data.h"
#include "svs/quantization/lvq/lvq.h"

// mkl
#include "mkl_lapacke.h"
#include <mkl.h>

// stl
#include <optional>
#include <string>
#include <string_view>
#include <variant>

// third-party
#include "fmt/core.h"

namespace svs {
namespace leanvec {

namespace detail {
template <typename T, size_t N, typename M, threads::ThreadPool Pool>
void remove_means(
    data::SimpleDataView<T, N> data, std::span<const M> means, Pool& threadpool
) {
    assert(data.dimensions() == means.size());
    threads::run(
        threadpool,
        threads::StaticPartition(data.size()),
        [&](auto&& is, uint64_t SVS_UNUSED(tid)) {
            size_t dim = data.dimensions();
            for (auto i : is) {
                auto datum = data.get_datum(i);
                for (size_t j = 0; j < dim; ++j) {
                    datum[j] -= means[j];
                }
            }
        }
    );
}
} // namespace detail

///
/// Transform in LeanVec using the precalculated matrix, this method is
/// single threaded suited for smaller dataset/query conversions. For, larger data
/// conversion use "transform_batch" method below
///
/// Requires ``data`` and ``leanvec_matrix`` to have dense in-memory representations.
///
template <size_t N1, size_t N2>
data::SimpleData<float> transform_leanvec(
    data::ConstSimpleDataView<float, N1> data,
    data::ConstSimpleDataView<float, N2> leanvec_matrix
) {
    assert(data.dimensions() == leanvec_matrix.size());

    size_t dims = data.dimensions();
    size_t leanvec_dims = leanvec_matrix.dimensions();
    data::SimpleData<float> leanvec_data{data.size(), leanvec_dims};

    // Performs C = alpha * A * B + beta * C
    // where
    // * A is a M x K matrix
    // * B is a K x N matrix
    // * C is a M x N matrix
    cblas_sgemm(
        CblasRowMajor,         // CBLAS_LAYOUT layout
        CblasNoTrans,          // CBLAS_TRANSPOSE TransA
        CblasNoTrans,          // CBLAS_TRANSPOSE TransB
        data.size(),           // const int M
        leanvec_dims,          // const int N
        dims,                  // const int K
        1.0,                   // float alpha
        data.data(),           // const float* A
        dims,                  // const int lda
        leanvec_matrix.data(), // const float* B
        leanvec_dims,          // const int ldb
        0.0,                   // const float beta
        leanvec_data.data(),   // float* c
        leanvec_dims           // const int ldc
    );

    return leanvec_data;
}

// Transform queries in LeanVec domain
// Need to subtract means for L2 queries
template <size_t N1, size_t N2>
data::SimpleData<float> transform_queries(
    const distance::DistanceL2& SVS_UNUSED(dist),
    data::ConstSimpleDataView<float, N1> queries,
    data::ConstSimpleDataView<float, N2> leanvec_matrix,
    std::span<const double> means
) {
    size_t dims = queries.dimensions();
    data::SimpleData<float> processed_queries{queries.size(), dims};

    for (size_t i = 0; i < queries.size(); i++) {
        processed_queries.set_datum(i, queries.get_datum(i));
        auto datum = processed_queries.get_datum(i);
        for (size_t j = 0; j < dims; j++) {
            datum[j] -= means[j];
        }
    }

    return transform_leanvec(processed_queries.cview(), leanvec_matrix);
}

template <size_t N1, size_t N2>
data::SimpleData<float> transform_queries(
    const distance::DistanceIP& SVS_UNUSED(dist),
    data::ConstSimpleDataView<float, N1> queries,
    data::ConstSimpleDataView<float, N2> leanvec_matrix,
    std::span<const double> SVS_UNUSED(means)
) {
    return transform_leanvec(queries, leanvec_matrix);
}

///
/// Compute LeanVec (using PCA) Matrix using a sample set of data
//
template <
    size_t Extent,
    size_t LeanVecDims,
    data::ImmutableMemoryDataset Dataset,
    threads::ThreadPool Pool>
data::SimpleData<float, LeanVecDims> compute_leanvec_matrix(
    const Dataset& data,
    const std::vector<double>& means,
    Pool& threadpool,
    lib::MaybeStatic<LeanVecDims> leanvec_dims
) {
    using T = typename Dataset::element_type;
    static_assert(
        std::is_same_v<T, Float16> || std::is_same_v<T, float>,
        "LeanVec conversion only supported for Float and Float16"
    );

    auto dims = data.dimensions();
    if (leanvec_dims > dims) {
        throw ANNEXCEPTION("Invalid LeanVec dimensions!");
    }

    auto leanvec_matrix = data::SimpleData<float, LeanVecDims>(dims, leanvec_dims);

    // TODO: parameterize sample_size
    // Samples used for computing LeanVec matrix
    size_t sample_size = std::min(data.size(), size_t(100'000));
    auto sample_data = data::SimpleData<double, Extent>(sample_size, dims);
    for (size_t i = 0; i < sample_size; i++) {
        sample_data.set_datum(i, data.get_datum(i));
    }

    // Subtract means from the sample data
    assert(dims == means.size());
    detail::remove_means(sample_data.view(), lib::as_const_span(means), threadpool);

    // SVD computation to obtain PCA matrix
    auto vt = Matrix<double>{make_dims(dims, dims)};
    auto superb = Matrix<double>{make_dims(sample_size, dims)};
    auto s = Vector<double>{dims};

    // TODO: Annotate.
    auto info = LAPACKE_dgesvd(
        LAPACK_ROW_MAJOR,
        'N',
        'A',
        sample_size,
        dims,
        sample_data.data(),
        dims,
        s.data(),
        NULL,
        sample_size,
        vt.data(),
        dims,
        superb.data()
    );

    if (info > 0) {
        throw ANNEXCEPTION("The algorithm computing SVD failed to converge!");
    }

    // Transpose the eigenvector matrix and reduce dimensionality to LeanVecDims
    mkl_dimatcopy('R', 'T', dims, dims, 1.0, vt.data(), dims, dims);
    for (size_t i = 0; i < dims; i++) {
        auto datum = vt.slice(i).first(leanvec_dims);
        leanvec_matrix.set_datum(i, datum);
    }

    return leanvec_matrix;
}

///
/// Convert to LeanVec using the provided matrix, efficient for larger datasets
/// as it uses threadpool to compute in smaller batches.
/// For PCA to work, use zero mean data (means are subtracted)
template <
    size_t Extent,
    size_t LeanVecDims,
    data::ImmutableMemoryDataset Dataset,
    threads::ThreadPool Pool,
    typename Alloc = lib::Allocator<std::byte>>
auto transform_batch(
    const Dataset& data,
    data::ConstSimpleDataView<float, LeanVecDims> leanvec_matrix,
    const std::vector<double>& means,
    Pool& threadpool,
    const Alloc& allocator_prototype = {},
    bool is_pca = true
) {
    using T = typename Dataset::element_type;
    static_assert(
        std::is_same_v<T, Float16> || std::is_same_v<T, float>,
        "LeanVec conversion only supported for Float and Float16"
    );

    auto dims = data.dimensions();
    auto leanvec_dims = lib::MaybeStatic<LeanVecDims>(leanvec_matrix.dimensions());
    if (leanvec_dims > dims) {
        throw ANNEXCEPTION("Invalid LeanVec dimensions!");
    }

    using allocator_type = lib::rebind_allocator_t<T, Alloc>;
    allocator_type rebound_alloctor{allocator_prototype};

    data::SimpleData<T, LeanVecDims, allocator_type> leanvec_data{
        data.size(), leanvec_dims, rebound_alloctor};

    // Convert the entire dataset in LeanVec domain using the leanvec_matrix.
    // This is done in batches to save the memory footprint
    size_t batch_size = std::min(data.size(), size_t(1000'000));
    auto batch_data = data::SimpleData<float, Extent>(batch_size, dims);
    auto leanvec_batch_data =
        data::SimpleData<float, LeanVecDims>(batch_size, leanvec_dims);

    for (size_t i = 0; i < data.size(); i += batch_size) {
        size_t batch_end = std::min(i + batch_size, data.size());
        size_t curr_batch_size = batch_end - i;

        for (size_t j = 0; j < curr_batch_size; j++) {
            batch_data.set_datum(j, data.get_datum(i + j));
        }

        if (is_pca) {
            // Zero mean the data
            detail::remove_means(batch_data.view(), lib::as_const_span(means), threadpool);
        }

        // MKL runs in sequential mode, using our native threads for parallelism
        threads::run(
            threadpool,
            threads::StaticPartition{curr_batch_size},
            [&](const auto& is, uint64_t SVS_UNUSED(tid)) {
                auto range = threads::UnitRange(is);

                cblas_sgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    range.size(),
                    leanvec_dims,
                    dims,
                    1.0,
                    // TODO: Implement better API for getting this pointer.
                    batch_data.get_datum(range.start()).data(),
                    dims,
                    leanvec_matrix.data(),
                    leanvec_dims,
                    0.0,
                    // TODO: Implement better API for getting this pointer.
                    leanvec_batch_data.get_datum(range.start()).data(),
                    leanvec_dims
                );
            }
        );

        // Write batch LeanVec data into LeanVec full data
        for (size_t j = 0; j < curr_batch_size; j++) {
            leanvec_data.set_datum(i + j, leanvec_batch_data.get_datum(j));
        }
    }

    return leanvec_data;
}

// Sentinel type to select an LVQ dataset as either the primary or secondary dataset
// for `LeanVec`.
template <size_t Bits> struct UsingLVQ {};

namespace detail {

template <typename T> inline constexpr bool is_using_lvq_tag_v = false;
template <size_t N> inline constexpr bool is_using_lvq_tag_v<UsingLVQ<N>> = true;

} // namespace detail

// The LeanVec dataset is designed to use combinations of uncompressed and LVQ-compressed
// datasets.
//
// Entries in the namespace below provide auxiliary support for these features.
namespace detail {

// Specialization point for picking the data structure that will hold data.
// The `Alloc` parameter should be an allocator with a value type of `std::byte`.
// The propery allocator type will be available in the `allocator_type` alias.
template <typename T, size_t Extent, typename Alloc> struct PickContainer;

// Simple data types are stored in `SimpleData`.
template <typename T, size_t Extent, typename Alloc>
    requires(svs::has_datatype_v<T>)
struct PickContainer<T, Extent, Alloc> {
    static_assert(std::is_same_v<lib::allocator_value_type_t<Alloc>, std::byte>);
    using allocator_type = lib::rebind_allocator_t<T, Alloc>;
    using type = svs::data::SimpleData<T, Extent, allocator_type>;
};

// LVQ dataset use a one-level LVQ dataset.
template <size_t N, size_t Extent, typename Alloc>
struct PickContainer<UsingLVQ<N>, Extent, Alloc> {
    static_assert(std::is_same_v<lib::allocator_value_type_t<Alloc>, std::byte>);
    using allocator_type = Alloc;
    using type =
        quantization::lvq::LVQDataset<N, 0, Extent, quantization::lvq::Sequential, Alloc>;
};

// Use Turbo-encoding for 4-bit LVQ.
template <size_t Extent, typename Alloc> struct PickContainer<UsingLVQ<4>, Extent, Alloc> {
    static_assert(std::is_same_v<lib::allocator_value_type_t<Alloc>, std::byte>);
    using allocator_type = Alloc;
    using strategy_type = quantization::lvq::Turbo<16, 8>;
    using type = quantization::lvq::LVQDataset<4, 0, Extent, strategy_type, Alloc>;
};

template <typename T, size_t Extent, typename Alloc>
using pick_container_t = typename PickContainer<T, Extent, Alloc>::type;

// Create a container for the given data.
// Construct a simple dataset
template <typename T, size_t Extent, typename Alloc, data::ImmutableMemoryDataset Data>
svs::data::SimpleData<T, Extent, Alloc> create_container(
    lib::Type<data::SimpleData<T, Extent, Alloc>> SVS_UNUSED(type),
    const Data& original,
    threads::ThreadPool auto& SVS_UNUSED(threadpool),
    size_t SVS_UNUSED(alignment),
    const Alloc& allocator
) {
    auto dst = svs::data::SimpleData<T, Extent, Alloc>(
        original.size(), original.dimensions(), allocator
    );
    svs::data::copy(original, dst);
    return dst;
}

// Construct an LVQ dataset.
template <quantization::lvq::IsLVQDataset As, data::ImmutableMemoryDataset Data>
As create_container(
    lib::Type<As> SVS_UNUSED(type),
    const Data& original,
    threads::ThreadPool auto& threadpool,
    size_t alignment,
    const typename As::allocator_type& allocator
) {
    return As::compress(original, threadpool, alignment, allocator);
}

///// Dataset Loading
// For simple data - drop the alignment argument.
template <typename T, size_t Extent, typename Alloc>
svs::data::SimpleData<T, Extent, Alloc> load_container(
    lib::Type<svs::data::SimpleData<T, Extent, Alloc>> SVS_UNUSED(type),
    const lib::LoadTable& table,
    std::string_view key,
    size_t SVS_UNUSED(alignment),
    const Alloc& allocator
) {
    return lib::load_at<svs::data::SimpleData<T, Extent, Alloc>>(table, key, allocator);
}

// Respect alignment for LVQ data.
template <quantization::lvq::IsLVQDataset As>
As load_container(
    lib::Type<As> SVS_UNUSED(type),
    const lib::LoadTable& table,
    std::string_view key,
    size_t alignment,
    const typename As::allocator_type& allocator
) {
    return lib::load_at<As>(table, key, alignment, allocator);
}

///// Distance adaptors
// Uncompressed: No need for adapting.
template <typename T, size_t N, typename Alloc, typename Distance>
Distance adapt_distance(
    const data::SimpleData<T, N, Alloc>& SVS_UNUSED(original), const Distance& distance
) {
    return threads::shallow_copy(distance);
}

template <typename T, size_t N, typename Alloc, typename Distance>
Distance adapt_distance_for_self(
    const data::SimpleData<T, N, Alloc>& SVS_UNUSED(original), const Distance& distance
) {
    return threads::shallow_copy(distance);
}

// LVQ: Route through `quantization::lvq::adapt` and friends.
template <quantization::lvq::IsLVQDataset Data, typename Distance>
auto adapt_distance(const Data& data, const Distance& distance) {
    return quantization::lvq::adapt(data, distance);
}
template <quantization::lvq::IsLVQDataset Data, typename Distance>
auto adapt_distance_for_self(const Data& data, const Distance& distance) {
    return quantization::lvq::adapt_for_self(data, distance);
}

///// Decompressors
template <typename T, size_t N, typename Alloc>
lib::identity make_decompressor(const data::SimpleData<T, N, Alloc>& SVS_UNUSED(data)) {
    return lib::identity();
}

template <quantization::lvq::IsLVQDataset Data> auto make_decompressor(const Data& data) {
    return data.decompressor();
}

} // namespace detail

// Compatible type parameters for LeanDatasets
template <typename T>
concept LeanCompatible = has_datatype_v<T> || detail::is_using_lvq_tag_v<T>;

namespace detail {
template <LeanCompatible T1, LeanCompatible T2> consteval bool check_parameters() {
    if (has_datatype_v<T1> && has_datatype_v<T2> && !std::is_same_v<T1, T2>) {
        throw ANNEXCEPTION("Uncompressed parameters are not the same");
    }
    return true;
}
} // namespace detail

// LeanVec matrices for transforming data and queries
template <size_t Extent> struct LeanVecMatrices {
  public:
    using leanvec_matrix_type = data::SimpleData<float, Extent>;

    LeanVecMatrices() = default;
    LeanVecMatrices(leanvec_matrix_type data_matrix, leanvec_matrix_type query_matrix)
        : data_matrix_{std::move(data_matrix)}
        , query_matrix_{std::move(query_matrix)} {
        // Check that the size and dimensionality of both the matrices should be same
        if (data_matrix_.size() != query_matrix_.size()) {
            throw ANNEXCEPTION("Mismatched data and query matrix sizes!");
        }
        if (data_matrix_.dimensions() != query_matrix_.dimensions()) {
            throw ANNEXCEPTION("Mismatched data and query matrix dimensions!");
        }
    }

    // This copy constructor is required for converting dynamic dimension matrices to
    // static dimension.
    template <size_t Dims>
    LeanVecMatrices(const LeanVecMatrices<Dims>& other)
        : data_matrix_{other.num_rows(), other.num_cols()}
        , query_matrix_{other.num_rows(), other.num_cols()} {
        // TODO: Maybe add a constructor or utility method to SimpleData to allow
        // safe down-casting to a static dimension.
        data::copy(other.view_data_matrix(), data_matrix_);
        data::copy(other.view_query_matrix(), query_matrix_);
    }

    // Return the number of rows in the matrices
    size_t num_rows() const { return data_matrix_.size(); };

    // Return the number of columns in the matrices
    size_t num_cols() const { return data_matrix_.dimensions(); };

    // Return cview of the data matrix
    auto view_data_matrix() const { return data_matrix_.cview(); };

    // Return cview of the query matrix
    auto view_query_matrix() const { return query_matrix_.cview(); };

    ///// IO
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
    static constexpr std::string_view serialization_schema = "leanvec_matrices";
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(data_matrix, ctx), SVS_LIST_SAVE_(query_matrix, ctx)}
        );
    }

    static LeanVecMatrices load(const lib::LoadTable& table) {
        return LeanVecMatrices{
            SVS_LOAD_MEMBER_AT_(table, data_matrix),
            SVS_LOAD_MEMBER_AT_(table, query_matrix)};
    }

  private:
    leanvec_matrix_type data_matrix_;
    leanvec_matrix_type query_matrix_;
};

// Hoist out schemas for reuse while auto-loading.
inline constexpr std::string_view lean_dataset_schema = "leanvec_dataset";
inline constexpr lib::Version lean_dataset_save_version = lib::Version(0, 0, 0);

// LeanVec Dataset
template <
    LeanCompatible T1,
    LeanCompatible T2,
    size_t LeanVecDims,
    size_t Extent,
    typename Alloc = lib::Allocator<std::byte>>
class LeanDataset {
  public:
    using alloc_traits = std::allocator_traits<Alloc>;
    static_assert(
        std::is_same_v<typename alloc_traits::value_type, std::byte>,
        "LeanVec Dataset requires that the value type of its allocator be std::byte."
    );
    static_assert(
        detail::check_parameters<T1, T2>(),
        "LeanVec parameters did not pass legality checks"
    );

    // top level allocator type. Use `std::byte` as its value type because it can vary
    // depending on what is selected as the internal containers.
    using allocator_type = Alloc;

    // Container Aliases.
    using primary_type = detail::pick_container_t<T1, LeanVecDims, allocator_type>;
    using secondary_type = detail::pick_container_t<T2, Extent, allocator_type>;

    // Allocator aliases
    using primary_allocator_type = typename primary_type::allocator_type;
    using secondary_allocator_type = typename secondary_type::allocator_type;

    ///// Access extent constants.

    /// @brief The compile-time extent of the inner LeanVec dimensions.
    ///
    /// Set to ``svs::Dynamic`` if the inner dimsensions are determined at runtime.
    static constexpr size_t leanvec_extent = LeanVecDims;

    /// @brief The compile-time extent of the full dataset.
    ///
    /// Set to ``svs::Dynamic`` if the inner dimsensions are determined at runtime.
    static constexpr size_t extent = Extent;

    using leanvec_matrices_type = LeanVecMatrices<LeanVecDims>;
    using primary_data_type = T1;
    using secondary_data_type = T2;

  private:
    primary_type primary_;
    secondary_type secondary_;
    // N.B. When used in PCA mode, the contents of both the data and query matrices are
    // identical. When not using PCA mode, the contents can be different corresponding to
    // the different transforms applied to the dataset and queries.
    leanvec_matrices_type matrices_;
    std::vector<double> means_;
    bool is_pca_;

  public:
    using value_type = data::value_type_t<primary_type>;
    using const_value_type = data::const_value_type_t<primary_type>;
    using secondary_const_value_type = data::const_value_type_t<secondary_type>;

    ///// Constructors
    LeanDataset(
        primary_type primary,
        secondary_type secondary,
        leanvec_matrices_type matrices,
        std::vector<double> means,
        bool is_pca
    )
        : primary_{std::move(primary)}
        , secondary_{std::move(secondary)}
        , matrices_{std::move(matrices)}
        , means_{std::move(means)}
        , is_pca_{is_pca} {
        if (primary_.dimensions() != matrices_.num_cols()) {
            throw ANNEXCEPTION("Leanvec matrix columns should match primary dimensions!");
        }
        if (secondary_.dimensions() != matrices_.num_rows()) {
            throw ANNEXCEPTION("Leanvec matrix rows should match secondary dimensions!");
        }
    }

    // Dataset API
    size_t size() const { return primary_.size(); }

    /// @brief Return the dimensions of the full-precision dataset.
    size_t dimensions() const { return secondary_.dimensions(); }

    /// @brief Return the dimensions of the reduced dataset.
    size_t inner_dimensions() const { return primary_.dimensions(); }

    /// @brief Default method accesses primary/LeanVec dataset.
    const_value_type get_datum(size_t i) const { return primary_.get_datum(i); }

    /// @brief Access secondary dataset.
    secondary_const_value_type get_secondary(size_t i) const {
        return secondary_.get_datum(i);
    }

    /// @brief Prefetch LeanVec dataset.
    void prefetch(size_t i) const { primary_.prefetch(i); }

    /// @brief Prefetch secondary dataset.
    void prefetch_secondary(size_t i) const { secondary_.prefetch(i); }

    ///// Inspecting Primary and Residual dataset.
    const primary_type& view_primary_dataset() const { return primary_; }
    const secondary_type& view_secondary_dataset() const { return secondary_; }

    ///// Insertion

    ///
    /// @brief Encode and insert the provided data into the dataset.
    ///
    /// This inserts datum into both the primary and secondary datasets.
    /// For the primary dataset, the datum is transformed by the transformation matrix
    /// and its dimensionality is reduced.
    ///
    template <typename U, size_t N> void set_datum(size_t i, std::span<U, N> datum) {
        auto dims = secondary_.dimensions();
        if (datum.size() != dims) {
            throw ANNEXCEPTION("set_datum dimension should match orginal data!");
        }

        // Use a temporary SimpleData vector to hold the datum
        auto buffer = data::SimpleData<float, N>{1, dims};
        auto temp_datum = buffer.get_datum(0);
        std::copy(datum.begin(), datum.end(), temp_datum.begin());

        // Subtract means for PCA transformation
        if (is_pca_ == true) {
            for (size_t i = 0; i < dims; ++i) {
                temp_datum[i] -= means_[i];
            }
        }

        auto leanvec_data = transform_leanvec(buffer.cview(), matrices_.view_data_matrix());

        primary_.set_datum(i, leanvec_data.get_datum(0));
        secondary_.set_datum(i, datum);
    }

    ///// Distance Adaptors
    template <typename Distance> auto adapt(const Distance& distance) const {
        return detail::adapt_distance(primary_, distance);
    }

    template <typename Distance> auto adapt_secondary(const Distance& distance) const {
        return detail::adapt_distance(secondary_, distance);
    }

    template <typename Distance> auto adapt_for_self(const Distance& distance) const {
        return detail::adapt_distance_for_self(primary_, distance);
    }

    ///
    /// @brief Return a copy-consructible accessor to decompress the primary dataset.
    ///
    /// When the primary dataset is in a compressed form (such as LVQ) - it is
    /// more efficient to pre-allocate extra state to assist in decompression.
    ///
    /// The return type must be copy-constructible so worker-threads can efficiently copy
    /// it when performing parallel processing.
    ///
    auto decompressor() const { return detail::make_decompressor(primary_); }

    ///
    /// @brief Transform a collection of queries using the transformation matrix.
    ///
    /// This function intentionally has a narrow-contract on the type of the supplied
    /// queries as we rely on the queries having a specific layout in memory.
    ///
    template <typename Distance, size_t N>
    data::SimpleData<float> preprocess_queries(
        const Distance& distance, data::ConstSimpleDataView<float, N> queries
    ) const {
        // In PCA L2 queries, need to subtract the means
        if (is_pca_ == true) {
            return transform_queries(
                distance, queries, matrices_.view_query_matrix(), means_
            );
        }
        return transform_leanvec(queries, matrices_.view_query_matrix());
    }

    ///// Static Constructors.

    // Reduce dimensionality using either PCA or the provide matrices
    template <data::ImmutableMemoryDataset Dataset>
    static LeanDataset reduce(
        const Dataset& data,
        size_t num_threads = 1,
        size_t alignment = 0,
        lib::MaybeStatic<LeanVecDims> leanvec_dims = {},
        const allocator_type& allocator = {}
    ) {
        return reduce(data, std::nullopt, num_threads, alignment, leanvec_dims, allocator);
    }

    template <data::ImmutableMemoryDataset Dataset>
    static LeanDataset reduce(
        const Dataset& data,
        std::optional<leanvec_matrices_type> matrices,
        size_t num_threads = 1,
        size_t alignment = 0,
        lib::MaybeStatic<LeanVecDims> leanvec_dims = {},
        const allocator_type& allocator = {}
    ) {
        auto pool = threads::NativeThreadPool{num_threads};
        return reduce(data, std::move(matrices), pool, alignment, leanvec_dims, allocator);
    }

    // TODO: Correctly handle r-value references to the original dataset to take ownership
    // of it as the secondary dataset if allowed and applicable.
    template <data::ImmutableMemoryDataset Dataset, threads::ThreadPool Pool>
    static LeanDataset reduce(
        const Dataset& data,
        std::optional<leanvec_matrices_type> matrices,
        Pool& threads,
        size_t alignment = 0,
        lib::MaybeStatic<LeanVecDims> leanvec_dims = {},
        const allocator_type& allocator = {}
    ) {
        leanvec_matrices_type leanvec_matrices{};
        bool is_pca = !matrices.has_value();

        std::vector<double> means = utils::compute_medioid(data, threads);
        if (is_pca) {
            // Compute the transformation matrix for (a subset of ) the dataset.
            auto matrix = compute_leanvec_matrix<Extent, LeanVecDims>(
                data, means, threads, leanvec_dims
            );
            leanvec_matrices = leanvec_matrices_type(matrix, matrix);
        } else {
            leanvec_matrices = std::move(matrices.value());
        }

        // Transform the original dataset.
        auto leanvec_data = transform_batch<Extent>(
            data, leanvec_matrices.view_data_matrix(), means, threads, allocator, is_pca
        );

        auto primary_allocator = primary_allocator_type{allocator};
        auto secondary_allocator = secondary_allocator_type{allocator};

        auto type_primary = lib::Type<primary_type>();
        auto type_secondary = lib::Type<secondary_type>();

        return LeanDataset{
            detail::create_container(
                type_primary, leanvec_data, threads, alignment, primary_allocator
            ),
            detail::create_container(
                type_secondary, data, threads, alignment, secondary_allocator
            ),
            std::move(leanvec_matrices),
            std::move(means),
            is_pca};
    }

    ///// IO
    static constexpr lib::Version save_version = lean_dataset_save_version;
    static constexpr std::string_view serialization_schema = lean_dataset_schema;
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(primary, ctx),
             SVS_LIST_SAVE_(secondary, ctx),
             SVS_LIST_SAVE_(matrices, ctx),
             SVS_LIST_SAVE_(means),
             SVS_LIST_SAVE_(is_pca)}
        );
    }

    static LeanDataset load(
        const lib::LoadTable& table,
        size_t alignment = 0,
        const allocator_type& allocator = {}
    ) {
        auto type_primary = lib::Type<primary_type>();
        auto type_secondary = lib::Type<secondary_type>();
        auto primary_allocator = primary_allocator_type{allocator};
        auto secondary_allocator = secondary_allocator_type{allocator};

        return LeanDataset{
            detail::load_container(
                type_primary, table, "primary", alignment, primary_allocator
            ),
            detail::load_container(
                type_secondary, table, "secondary", alignment, secondary_allocator
            ),
            SVS_LOAD_MEMBER_AT_(table, matrices),
            SVS_LOAD_MEMBER_AT_(table, means),
            SVS_LOAD_MEMBER_AT_(table, is_pca)};
    }
};

/////
///// LeanDataset Concept
/////

namespace detail {
template <typename T> inline constexpr bool is_leanvec_dataset = false;
template <
    LeanCompatible T1,
    LeanCompatible T2,
    size_t LeanVecDims,
    size_t Extent,
    typename Alloc>
inline constexpr bool is_leanvec_dataset<LeanDataset<T1, T2, LeanVecDims, Extent, Alloc>> =
    true;
} // namespace detail

template <typename T>
concept IsLeanDataset = detail::is_leanvec_dataset<T>;

/////
///// Load Helpers
/////

// Types to use for leanvec.
inline constexpr lib::Types<float, Float16> LeanVecSourceTypes{};

// LeanVec based loaders can either perform LeanVec conversion online, or reload a
// previously saved LeanVec dataset.
struct OnlineLeanVec {
  public:
    explicit OnlineLeanVec(const std::filesystem::path& path, DataType type)
        : path{path}
        , type{type} {
        if (!lib::in(type, LeanVecSourceTypes)) {
            throw ANNEXCEPTION("Invalid type!");
        }
    }

    // Members
  public:
    std::filesystem::path path;
    DataType type;
};

struct Reload {
  public:
    explicit Reload(const std::filesystem::path& directory)
        : directory{directory} {}

    // Members
  public:
    std::filesystem::path directory;
};

// The various ways we can instantiate LeanVec-based datasets..
using SourceTypes = std::variant<OnlineLeanVec, Reload>;

/// A type used to request a specific specialization of LeanVec at runtime.
/// Used for dispatching.
enum class LeanVecKind { float32, float16, lvq8, lvq4 };

namespace detail {

template <LeanCompatible T> struct LeanVecPicker;

template <> struct LeanVecPicker<float> {
    static constexpr LeanVecKind value = LeanVecKind::float32;
};
template <> struct LeanVecPicker<svs::Float16> {
    static constexpr LeanVecKind value = LeanVecKind::float16;
};
template <> struct LeanVecPicker<UsingLVQ<8>> {
    static constexpr LeanVecKind value = LeanVecKind::lvq8;
};
template <> struct LeanVecPicker<UsingLVQ<4>> {
    static constexpr LeanVecKind value = LeanVecKind::lvq4;
};

} // namespace detail

template <typename T>
inline constexpr LeanVecKind leanvec_kind_v = detail::LeanVecPicker<T>::value;

// LeanDataset Matcher
struct Matcher {
  private:
    struct DatasetLayout {
        size_t dims;
        LeanVecKind kind;
    };

    static lib::TryLoadResult<DatasetLayout>
    detect_data(const lib::ContextFreeNodeView<toml::node>& node) {
        // Is it an uncompressed dataset?
        auto maybe_uncompressed = lib::try_load<svs::data::Matcher>(node);
        auto failure = lib::Unexpected{lib::TryLoadFailureReason::Other};

        // On success - determine if this one of the recognized types.
        if (maybe_uncompressed) {
            const auto& matcher = maybe_uncompressed.value();
            size_t dims = matcher.dims;
            switch (matcher.eltype) {
                case DataType::float16: {
                    return DatasetLayout{dims, LeanVecKind::float16};
                }
                case DataType::float32: {
                    return DatasetLayout{dims, LeanVecKind::float32};
                }
                default: {
                    return failure;
                }
            }
        }

        // Failed to match the uncompressed layout. Try LVQ.
        auto maybe_lvq = lib::try_load<svs::quantization::lvq::Matcher>(node);
        if (maybe_lvq) {
            const auto& matcher = maybe_lvq.value();
            size_t dims = matcher.dims;
            size_t primary = matcher.primary;
            switch (primary) {
                case 4: {
                    return DatasetLayout{dims, LeanVecKind::lvq4};
                }
                case 8: {
                    return DatasetLayout{dims, LeanVecKind::lvq8};
                }
                default: {
                    return failure;
                }
            }
        }
        return lib::Unexpected(lib::TryLoadFailureReason::InvalidSchema);
    }

  public:
    ///// Loading.
    static bool check_load_compatibility(std::string_view schema, lib::Version version) {
        return schema == lean_dataset_schema && version == lean_dataset_save_version;
    }

    static lib::TryLoadResult<Matcher> try_load(const lib::ContextFreeLoadTable& table) {
        // For each of the primary and secondary, use the combinations of expected expected
        // types until we have a successful match.
        auto primary_expected = detect_data(table.at("primary"));
        if (!primary_expected) {
            return lib::Unexpected(primary_expected.error());
        }

        auto secondary_expected = detect_data(table.at("secondary"));
        if (!secondary_expected) {
            return lib::Unexpected(secondary_expected.error());
        }

        const auto& primary = primary_expected.value();
        const auto& secondary = secondary_expected.value();

        return Matcher{
            .leanvec_dims = primary.dims,
            .total_dims = secondary.dims,
            .primary_kind = primary.kind,
            .secondary_kind = secondary.kind};
    }

    static Matcher load(const lib::ContextFreeLoadTable& table) {
        // For each of the primary and secondary, use the combinations of expected expected
        // types until we have a successful match.
        auto primary_expected = detect_data(table.at("primary"));
        if (!primary_expected) {
            throw ANNEXCEPTION("Could not match the primary dataset!");
        }

        auto secondary_expected = detect_data(table.at("secondary"));
        if (!secondary_expected) {
            throw ANNEXCEPTION("Could not match the secondary dataset!");
        }

        const auto& primary = primary_expected.value();
        const auto& secondary = secondary_expected.value();

        return Matcher{
            .leanvec_dims = primary.dims,
            .total_dims = secondary.dims,
            .primary_kind = primary.kind,
            .secondary_kind = secondary.kind};
    }

    constexpr bool friend operator==(const Matcher&, const Matcher&) = default;

    ///// Members
    size_t leanvec_dims;
    size_t total_dims;
    LeanVecKind primary_kind;
    LeanVecKind secondary_kind;
};

// Overload Matching Rules
template <LeanCompatible T1, LeanCompatible T2, size_t LeanVecDims, size_t Extent>
int64_t overload_score(
    LeanVecKind primary, size_t primary_dims, LeanVecKind secondary, size_t secondary_dims
) {
    // Check primary kind
    if (primary != leanvec::leanvec_kind_v<T1>) {
        return lib::invalid_match;
    }

    // Check secondary kind
    if (secondary != leanvec::leanvec_kind_v<T2>) {
        return lib::invalid_match;
    }

    // Check extent-tags.
    auto extent_match = lib::dispatch_match<lib::ExtentArg, lib::ExtentTag<Extent>>(
        lib::ExtentArg{secondary_dims}
    );

    // If extents don't match, then we abort immediately.
    if (extent_match < 0) {
        return lib::invalid_match;
    }

    // Check leanvec_dims-tags.
    auto leanvec_dims_match =
        lib::dispatch_match<lib::ExtentArg, lib::ExtentTag<LeanVecDims>>(lib::ExtentArg{
            primary_dims});

    // If leanvec_dims don't match, then we abort immediately.
    if (leanvec_dims_match < 0) {
        return lib::invalid_match;
    }

    return extent_match + leanvec_dims_match;
}

template <LeanCompatible T1, LeanCompatible T2, size_t LeanVecDims, size_t Extent>
int64_t overload_score(const Matcher& matcher) {
    return overload_score<T1, T2, LeanVecDims, Extent>(
        matcher.primary_kind,
        matcher.leanvec_dims,
        matcher.secondary_kind,
        matcher.total_dims
    );
}

// Forward Declaration.
template <typename T1, typename T2, size_t LeanVecDims, size_t Extent, typename Alloc>
struct LeanVecLoader;

template <typename Alloc = lib::Allocator<std::byte>> struct ProtoLeanVecLoader {
  public:
    ProtoLeanVecLoader() = default;
    explicit ProtoLeanVecLoader(
        const UnspecializedVectorDataLoader<Alloc>& datafile,
        size_t leanvec_dims,
        LeanVecKind primary_kind,
        LeanVecKind secondary_kind,
        std::optional<LeanVecMatrices<Dynamic>> matrices,
        size_t alignment = 0
    )
        : source_{std::in_place_type<OnlineLeanVec>, datafile.path_, datafile.type_}
        , leanvec_dims_{leanvec_dims}
        , dims_{datafile.dims_}
        , primary_kind_{primary_kind}
        , secondary_kind_{secondary_kind}
        , matrices_{std::move(matrices)}
        , alignment_{alignment}
        , allocator_{datafile.allocator_} {}

    explicit ProtoLeanVecLoader(
        Reload reloader,
        // size_t leanvec_dims,
        // size_t dims,
        // LeanVecKind primary_kind,
        // LeanVecKind secondary_kind,
        size_t alignment = 0,
        const Alloc& allocator = {}
    )
        : source_{std::move(reloader)}
        , matrices_{std::nullopt}
        , alignment_{alignment}
        , allocator_{allocator} {
        // Produce a hard error if we cannot load and match the dataset.
        auto matcher = lib::load_from_disk<Matcher>(std::get<Reload>(source_).directory);
        primary_kind_ = matcher.primary_kind;
        secondary_kind_ = matcher.secondary_kind;
        leanvec_dims_ = matcher.leanvec_dims;
        dims_ = matcher.total_dims;
    }

    template <
        typename T1,
        typename T2,
        size_t LeanVecDims,
        size_t Extent,
        typename F = std::identity>
    LeanVecLoader<
        T1,
        T2,
        LeanVecDims,
        Extent,
        std::decay_t<std::invoke_result_t<F, const Alloc&>>>
    refine(lib::Val<Extent>, F&& f = std::identity()) const {
        using ARet = std::decay_t<std::invoke_result_t<F, const Alloc&>>;
        // Make sure the pre-set values are correct.
        if constexpr (Extent != Dynamic) {
            if (Extent != dims_) {
                throw ANNEXCEPTION("Invalid Extent specialization!");
            }
        }

        if constexpr (LeanVecDims != Dynamic) {
            if (LeanVecDims != leanvec_dims_) {
                throw ANNEXCEPTION("Invalid LeanVecDims specialization!");
            }
        }

        if (leanvec_kind_v<T1> != primary_kind_) {
            throw ANNEXCEPTION("Invalid Primary kind specialization!");
        }

        if (leanvec_kind_v<T2> != secondary_kind_) {
            throw ANNEXCEPTION("Invalid Secondary kind specialization!");
        }

        // Convert dynamic Extent matrices to static LeanVecDims
        auto matrices = std::optional<LeanVecMatrices<LeanVecDims>>(matrices_);

        return LeanVecLoader<T1, T2, LeanVecDims, Extent, ARet>(
            source_, leanvec_dims_, std::move(matrices), alignment_, f(allocator_)
        );
    }

  public:
    SourceTypes source_;
    size_t leanvec_dims_;
    size_t dims_;
    LeanVecKind primary_kind_;
    LeanVecKind secondary_kind_;
    std::optional<LeanVecMatrices<Dynamic>> matrices_;
    size_t alignment_;
    Alloc allocator_;
};

template <typename T1, typename T2, size_t LeanVecDims, size_t Extent, typename Alloc>
struct LeanVecLoader {
  public:
    using loaded_type = LeanDataset<T1, T2, LeanVecDims, Extent, Alloc>;

    explicit LeanVecLoader(
        SourceTypes source,
        size_t leanvec_dims,
        std::optional<LeanVecMatrices<LeanVecDims>> matrices,
        size_t alignment,
        const Alloc& allocator
    )
        : source_{std::move(source)}
        , leanvec_dims_{leanvec_dims}
        , matrices_{std::move(matrices)}
        , alignment_{alignment}
        , allocator_{allocator} {}

    loaded_type load() const {
        auto pool = threads::SequentialThreadPool();
        return load(pool);
    }

    template <typename F>
    LeanVecLoader<
        T1,
        T2,
        LeanVecDims,
        Extent,
        std::decay_t<std::invoke_result_t<F, const Alloc&>>>
    rebind_alloc(const F& f) {
        return LeanVecLoader<
            T1,
            T2,
            LeanVecDims,
            Extent,
            std::decay_t<std::invoke_result_t<F, const Alloc&>>>{
            source_, leanvec_dims_, matrices_, alignment_, f(allocator_)};
    }

    template <threads::ThreadPool Pool> loaded_type load(Pool& threadpool) const {
        return std::visit<loaded_type>(
            [&](auto source) {
                using U = std::decay_t<decltype(source)>;
                if constexpr (std::is_same_v<U, Reload>) {
                    return lib::load_from_disk<loaded_type>(
                        source.directory, alignment_, allocator_
                    );
                } else {
                    return lib::match(
                        LeanVecSourceTypes,
                        source.type,
                        [&]<typename V>(lib::Type<V> SVS_UNUSED(type)) {
                            using rebind_type = lib::rebind_allocator_t<V, Alloc>;
                            return loaded_type::reduce(
                                data::SimpleData<V, Extent, rebind_type>::load(source.path),
                                matrices_,
                                threadpool,
                                alignment_,
                                leanvec_dims_,
                                allocator_
                            );
                        }
                    );
                }
            },
            source_
        );
    }

  private:
    SourceTypes source_;
    lib::MaybeStatic<LeanVecDims> leanvec_dims_;
    std::optional<LeanVecMatrices<LeanVecDims>> matrices_;
    size_t alignment_;
    Alloc allocator_;
};

} // namespace leanvec

// Define dispatch conversion from ProtoLeanVecLoader to LeanVecLoader.
template <
    typename Primary,
    typename Secondary,
    size_t LeanVecDims,
    size_t Extent,
    typename Alloc>
struct lib::DispatchConverter<
    leanvec::ProtoLeanVecLoader<Alloc>,
    leanvec::LeanVecLoader<Primary, Secondary, LeanVecDims, Extent, Alloc>> {
    static int64_t match(const leanvec::ProtoLeanVecLoader<Alloc>& loader) {
        return overload_score<Primary, Secondary, LeanVecDims, Extent>(
            loader.primary_kind_, loader.leanvec_dims_, loader.secondary_kind_, loader.dims_
        );
        // if (loader.primary_kind_ != leanvec::leanvec_kind_v<Primary>) {
        //     return lib::invalid_match;
        // }

        // // Check secondary kind
        // if (loader.secondary_kind_ != leanvec::leanvec_kind_v<Secondary>) {
        //     return lib::invalid_match;
        // }

        // // Check extent-tags.
        // auto extent_match = lib::dispatch_match<lib::ExtentArg, lib::ExtentTag<Extent>>(
        //     lib::ExtentArg{loader.dims_}
        // );

        // // If extents don't match, then we abort immediately.
        // if (extent_match < 0) {
        //     return lib::invalid_match;
        // }

        // // Check leanvec_dims-tags.
        // auto leanvec_dims_match =
        //     lib::dispatch_match<lib::ExtentArg,
        //     lib::ExtentTag<LeanVecDims>>(lib::ExtentArg{ loader.leanvec_dims_});
        // // If leanvec_dims don't match, then we abort immediately.
        // if (leanvec_dims_match < 0) {
        //     return lib::invalid_match;
        // }

        // return extent_match + leanvec_dims_match;
    }

    static leanvec::LeanVecLoader<Primary, Secondary, LeanVecDims, Extent, Alloc>
    convert(const leanvec::ProtoLeanVecLoader<Alloc>& loader) {
        return loader.template refine<Primary, Secondary, LeanVecDims, Extent>(
            lib::Val<Extent>()
        );
    }

    static std::string description() {
        auto dims = []() {
            if constexpr (Extent == Dynamic) {
                return "any";
            } else {
                return Extent;
            }
        }();

        auto leanvec_dims = []() {
            if constexpr (LeanVecDims == Dynamic) {
                return "any";
            } else {
                return LeanVecDims;
            }
        }();

        return fmt::format("LeanVecLoader dims-{}x{}", dims, leanvec_dims);
    }
};

} // namespace svs
