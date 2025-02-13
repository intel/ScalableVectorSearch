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
#include "svs/core/data.h"
#include "svs/quantization/lvq/lvq.h"

// Intel(R) MKL
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

// Hoist out schemas for reuse while auto-loading.
inline constexpr std::string_view lean_dataset_schema = "leanvec_dataset";
inline constexpr lib::Version lean_dataset_save_version = lib::Version(0, 0, 0);

// Sentinel type to select an LVQ dataset as either the primary or secondary
// dataset for `LeanVec`.
template <size_t Bits> struct UsingLVQ {};

namespace detail {

template <typename T> inline constexpr bool is_using_lvq_tag_v = false;
template <size_t N> inline constexpr bool is_using_lvq_tag_v<UsingLVQ<N>> = true;

// Used to SFINAE away resizing methods if the allocator is not blocked.
template <typename A> inline constexpr bool is_blocked = false;
template <typename A> inline constexpr bool is_blocked<data::Blocked<A>> = true;

template <typename A, bool = is_blocked<A>> struct select_underlying_allocator {
    using type = A;
};
template <typename A> struct select_underlying_allocator<A, true> {
    using type = typename A::allocator_type;
};
template <typename A>
using select_underlying_allocator_t = typename select_underlying_allocator<A>::type;

template <typename T, typename A, bool = is_blocked<A>> struct select_rebind_allocator {
    using type = lib::rebind_allocator_t<T, A>;
};
template <typename T, typename A> struct select_rebind_allocator<T, A, true> {
    using base_allocator = typename A::allocator_type;
    using rebind_base_allocator = lib::rebind_allocator_t<T, base_allocator>;
    using type = data::Blocked<rebind_base_allocator>;
};
template <typename T, typename A>
using select_rebind_allocator_t = typename select_rebind_allocator<T, A>::type;

// Specialization point for picking the data structure that will hold data.
// The `Alloc` parameter should be an allocator with a value type of
// `std::byte`. The propery allocator type will be available in the
// `allocator_type` alias.
template <typename T, size_t Extent, typename Alloc> struct PickContainer;

// Simple data types are stored in `SimpleData`.
template <typename T, size_t Extent, typename Alloc>
    requires(svs::has_datatype_v<T>)
struct PickContainer<T, Extent, Alloc> {
    static_assert(std::is_same_v<
                  lib::allocator_value_type_t<select_underlying_allocator_t<Alloc>>,
                  std::byte>);
    using allocator_type = select_rebind_allocator_t<T, Alloc>;
    using type = svs::data::SimpleData<T, Extent, allocator_type>;
};

// LVQ dataset use a one-level LVQ dataset.
template <size_t N, size_t Extent, typename Alloc>
struct PickContainer<UsingLVQ<N>, Extent, Alloc> {
    static_assert(std::is_same_v<
                  lib::allocator_value_type_t<select_underlying_allocator_t<Alloc>>,
                  std::byte>);
    using allocator_type = Alloc;
    using type =
        quantization::lvq::LVQDataset<N, 0, Extent, quantization::lvq::Sequential, Alloc>;
};

// Use Turbo-encoding for 4-bit LVQ.
template <size_t Extent, typename Alloc> struct PickContainer<UsingLVQ<4>, Extent, Alloc> {
    static_assert(std::is_same_v<
                  lib::allocator_value_type_t<select_underlying_allocator_t<Alloc>>,
                  std::byte>);
    using allocator_type = Alloc;
    using strategy_type = quantization::lvq::Turbo<16, 8>;
    using type = quantization::lvq::LVQDataset<4, 0, Extent, strategy_type, Alloc>;
};

template <typename T, size_t Extent, typename Alloc>
using pick_container_t = typename PickContainer<T, Extent, Alloc>::type;

} // namespace detail

// Compatible type parameters for LeanDatasets
template <typename T>
concept LeanCompatible = has_datatype_v<T> || detail::is_using_lvq_tag_v<T>;

// Trait to determine if an allocator is blocked or not.
template <typename A>
concept is_resizeable = detail::is_blocked<A>;

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
    LeanVecMatrices(leanvec_matrix_type data_matrix, leanvec_matrix_type query_matrix);
    template <size_t Dims> LeanVecMatrices(const LeanVecMatrices<Dims>& other);
    size_t num_rows() const;
    size_t num_cols() const;
    auto view_data_matrix() const;
    auto view_query_matrix() const;

    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
    static constexpr std::string_view serialization_schema = "leanvec_matrices";
    lib::SaveTable save(const lib::SaveContext& ctx) const;

    static LeanVecMatrices load(const lib::LoadTable& table);

  private:
    leanvec_matrix_type data_matrix_;
    leanvec_matrix_type query_matrix_;
};

// LeanVec Dataset
template <
    LeanCompatible T1,
    LeanCompatible T2,
    size_t LeanVecDims,
    size_t Extent,
    typename Alloc = lib::Allocator<std::byte>>
class LeanDataset {
  public:
    static_assert(
        std::is_same_v<
            lib::allocator_value_type_t<detail::select_underlying_allocator_t<Alloc>>,
            std::byte>,
        "LeanVec Dataset requires that the value type of its allocator "
        "be std::byte."
    );
    static_assert(
        detail::check_parameters<T1, T2>(),
        "LeanVec parameters did not pass legality checks"
    );

    // top level allocator type. Use `std::byte` as its value type because it can
    // vary depending on what is selected as the internal containers.
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
    /// Set to ``svs::Dynamic`` if the inner dimsensions are determined at
    /// runtime.
    static constexpr size_t leanvec_extent = LeanVecDims;

    /// @brief The compile-time extent of the full dataset.
    ///
    /// Set to ``svs::Dynamic`` if the inner dimsensions are determined at
    /// runtime.
    static constexpr size_t extent = Extent;

    using leanvec_matrices_type = LeanVecMatrices<LeanVecDims>;
    using primary_data_type = T1;
    using secondary_data_type = T2;

    using value_type = data::value_type_t<primary_type>;
    using const_value_type = data::const_value_type_t<primary_type>;
    using secondary_const_value_type = data::const_value_type_t<secondary_type>;

    LeanDataset(
        primary_type primary,
        secondary_type secondary,
        leanvec_matrices_type matrices,
        std::vector<double> means,
        bool is_pca
    );

    /// @brief Return the number of vectors in the dataset.
    size_t size() const;
    /// @brief Return the dimensions of the full-dimension dataset.
    size_t dimensions() const;
    /// @brief Return the dimensions of the reduced dataset.
    size_t inner_dimensions() const;
    /// @brief Default method accesses primary/LeanVec dataset.
    const_value_type get_datum(size_t i) const;
    /// @brief Access secondary dataset.
    secondary_const_value_type get_secondary(size_t i) const;
    /// @brief Prefetch LeanVec dataset.
    void prefetch(size_t i) const;
    /// @brief Prefetch secondary dataset.
    void prefetch_secondary(size_t i) const;

    /// Inspecting Primary and Residual dataset.
    const primary_type& view_primary_dataset() const;
    const secondary_type& view_secondary_dataset() const;

    ///// Insertion

    ///
    /// @brief Encode and insert the provided data into the dataset.
    ///
    /// This inserts datum into both the primary and secondary datasets.
    /// For the primary dataset, the datum is transformed by the transformation
    /// matrix and its dimensionality is reduced.
    ///
    template <typename U, size_t N> void set_datum(size_t i, std::span<U, N> datum);

    template <typename Data, typename Distance>
    Distance adapt(const Data& data, const Distance& distance) const;

    template <typename Data, typename Distance>
    Distance adapt_secondary(const Data& data, const Distance& distance) const;

    template <typename Data, typename Distance>
    Distance adapt_for_self(const Data& data, const Distance& distance) const;

    template <quantization::lvq::IsLVQDataset Data, typename Distance>
    quantization::lvq::biased_distance_t<Distance>
    adapt(const Data& SVS_UNUSED(data), const Distance& distance) const;

    template <quantization::lvq::IsLVQDataset Data, typename Distance>
    quantization::lvq::biased_distance_t<Distance>
    adapt_secondary(const Data& SVS_UNUSED(data), const Distance& distance) const;

    template <quantization::lvq::IsLVQDataset Data, typename Distance>
    quantization::lvq::DecompressionAdaptor<quantization::lvq::biased_distance_t<Distance>>
    adapt_for_self(const Data& SVS_UNUSED(data), const Distance& distance) const;

    ///
    /// @brief Return a copy-consructible accessor to decompress the primary
    /// dataset.
    ///
    /// When the primary dataset is in a compressed form (such as LVQ) - it is
    /// more efficient to pre-allocate extra state to assist in decompression.
    ///
    /// The return type must be copy-constructible so worker-threads can
    /// efficiently copy it when performing parallel processing.
    ///
    auto decompressor() const;

    ///
    /// @brief Transform a collection of queries using the transformation matrix.
    ///
    /// This function intentionally has a narrow-contract on the type of the
    /// supplied queries as we rely on the queries having a specific layout in
    /// memory.
    ///
    template <typename Distance, size_t N>
    data::SimpleData<float> preprocess_queries(
        const Distance& distance, data::ConstSimpleDataView<float, N> queries
    ) const;

    template <data::ImmutableMemoryDataset Dataset>
    static LeanDataset reduce(
        const Dataset& data,
        size_t num_threads = 1,
        size_t alignment = 0,
        lib::MaybeStatic<LeanVecDims> leanvec_dims = {},
        const allocator_type& allocator = {}
    );
    template <data::ImmutableMemoryDataset Dataset>
    static LeanDataset reduce(
        const Dataset& data,
        std::optional<leanvec_matrices_type> matrices,
        size_t num_threads = 1,
        size_t alignment = 0,
        lib::MaybeStatic<LeanVecDims> leanvec_dims = {},
        const allocator_type& allocator = {}
    );
    template <data::ImmutableMemoryDataset Dataset, threads::ThreadPool Pool>
    static LeanDataset reduce(
        const Dataset& data,
        std::optional<leanvec_matrices_type> matrices,
        Pool& threads,
        size_t alignment = 0,
        lib::MaybeStatic<LeanVecDims> leanvec_dims = {},
        const allocator_type& allocator = {}
    );

    void resize(size_t new_size)
        requires is_resizeable<Alloc>;

    template <std::integral I, threads::ThreadPool Pool>
    void
    compact(std::span<const I> new_to_old, Pool& threadpool, size_t batchsize = 1'000'000)
        requires is_resizeable<Alloc>;

    static constexpr lib::Version save_version = lean_dataset_save_version;
    static constexpr std::string_view serialization_schema = lean_dataset_schema;
    lib::SaveTable save(const lib::SaveContext& ctx) const;

    static LeanDataset load(
        const lib::LoadTable& table,
        size_t alignment = 0,
        const allocator_type& allocator = {}
    );

  private:
    primary_type primary_;
    secondary_type secondary_;
    // N.B. When used in PCA mode, the contents of both the data and query
    // matrices are identical. When not using PCA mode, the contents can be
    // different corresponding to the different transforms applied to the dataset
    // and queries.
    leanvec_matrices_type matrices_;
    std::vector<double> means_;
    bool is_pca_;
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

} // namespace leanvec
} // namespace svs
