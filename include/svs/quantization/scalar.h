/*
 * Copyright 2025 Intel Corporation
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

#include "eve/algo.hpp"

// svs
#include "svs/core/data/simple.h"
#include "svs/core/distance.h"
#include "svs/lib/memory.h"
#include "svs/lib/static.h"
#include "svs/lib/version.h"

// stl
#include <memory>

namespace svs {
namespace quantization {
namespace scalar {

class EuclideanCompressed {
  public:
    using compare = std::less<>;

    static constexpr bool implicit_broadcast = false;
    static constexpr bool must_fix_argument = true;

    EuclideanCompressed(float scale, float bias, size_t dims)
        : query_fp32_{1, dims}
        , scale_{scale}
        , bias_{bias}
        , scale_sq_{scale * scale} {}

    EuclideanCompressed shallow_copy() const {
        return EuclideanCompressed(scale_, bias_, query_fp32_.dimensions());
    }

    template <typename T> void fix_argument(const std::span<T>& query) {
        query_fp32_.set_datum(0, query);
    }

    std::span<const float> view_query() const { return query_fp32_.get_datum(0); }

    template <typename T>
    float compute(const T& y) const
        requires(lib::is_spanlike_v<T>)
    {
        auto inner = distance::DistanceL2{};
        return scale_sq_ * distance::compute(inner, view_query(), y);
    }

  private:
    data::SimpleData<float> query_fp32_;
    float scale_;
    float bias_;

    float scale_sq_;
};

class InnerProductCompressed {
  public:
    using compare = std::greater<>;

    static constexpr bool implicit_broadcast = false;
    static constexpr bool must_fix_argument = true;

    InnerProductCompressed(float scale, float bias, size_t dims)
        : query_fp32_{1, dims}
        , scale_{scale}
        , bias_{bias}
        , scale_sq_{scale * scale} {}

    InnerProductCompressed shallow_copy() const {
        return InnerProductCompressed(scale_, bias_, query_fp32_.dimensions());
    }

    template <typename T> void fix_argument(const std::span<T>& query) {
        query_fp32_.set_datum(0, query);

        float sum = eve::algo::reduce(query, 0.0f);
        offset_ = bias_ * scale_ * sum;
    }

    std::span<const float> view_query() const { return query_fp32_.get_datum(0); }

    template <typename T>
    float compute(const T& y) const
        requires(lib::is_spanlike_v<T>)
    {
        auto inner = distance::DistanceIP{};
        return scale_sq_ * distance::compute(inner, view_query(), y) + offset_;
    }

  private:
    data::SimpleData<float> query_fp32_;
    float scale_;
    float bias_;

    // pre-computed values
    float scale_sq_;
    float offset_ = 0;
};

class CosineSimilarityCompressed {
  public:
    using compare = std::greater<>;

    static constexpr bool implicit_broadcast = false;
    static constexpr bool must_fix_argument = true;

    CosineSimilarityCompressed(float scale, float bias, size_t dims)
        : query_fp32_{1, dims}
        , scale_{scale}
        , bias_{bias} {}

    template <typename T> void fix_argument(const std::span<T>& query) {
        query_fp32_.set_datum(0, query);
        inner_.fix_argument(query);
    }

    std::span<const float> view_query() const { return query_fp32_.get_datum(0); }

    template <typename T>
    float compute(const T& y) const
        requires(lib::is_spanlike_v<T>)
    {
        std::vector<float> y_biased(y.size());
        std::transform(y.begin(), y.end(), y_biased.begin(), [&](float v) {
            return (v - bias_) / scale_;
        });
        return distance::compute(inner_, view_query(), std::span<const float>(y_biased));
    }

  private:
    data::SimpleData<float> query_fp32_;
    float scale_;
    float bias_;

    distance::DistanceCosineSimilarity inner_;
};

namespace detail {

// Map from baseline distance functors to the local versions.
template <typename T> struct CompressedDistance;

template <> struct CompressedDistance<distance::DistanceL2> {
    using type = EuclideanCompressed;
};

template <> struct CompressedDistance<distance::DistanceIP> {
    using type = InnerProductCompressed;
};

template <> struct CompressedDistance<distance::DistanceCosineSimilarity> {
    using type = CosineSimilarityCompressed;
};

} // namespace detail

template <typename T>
using compressed_distance_t = typename detail::CompressedDistance<T>::type;

inline constexpr std::string_view scalar_quantization_serialization_schema =
    "scalar_quantization_dataset";
inline constexpr lib::Version scalar_quantization_save_version = lib::Version(0, 0, 0);

// Scalar Quantization Dataset
// This class provides a globally quantized (scale & bias) dataset.
template <size_t Extent = svs::Dynamic, typename Alloc = lib::Allocator<std::int8_t>>
class SQDataset {
  public:
    constexpr static size_t extent = Extent;
    constexpr static bool uses_compressed_data = true;

    using allocator_type = Alloc;
    // TODO: replace int8 with template
    using data_type = data::SimpleData<std::int8_t, Extent, allocator_type>;

    // TODO: get_datum will return this type, other classes would return compressed data
    //       while we return uncompressed data for simplicity. Maybe this needs to change
    // using const_value_type = std::span<const std::int8_t, Extent>;
    // using value_type = const_value_type;
    // TODO: This is potentially a performance bottleneck. Other datasets simply return a
    // view, but because we are manipulating the values before return, they must go into a
    // vector
    using const_value_type = std::span<const std::int8_t, Extent>;
    using value_type = const_value_type;
    using element_type = std::int8_t;

  private:
    float scale_;
    float bias_;
    data_type data_;

  public:
    SQDataset(size_t size, size_t dims)
        : data_{size, dims} {}
    SQDataset(data_type data, float scale, float bias)
        : scale_(scale)
        , bias_(bias)
        , data_{std::move(data)} {}

    size_t size() const { return data_.size(); }
    size_t dimensions() const { return data_.dimensions(); }

    float get_scale() const { return scale_; }
    float get_bias() const { return bias_; }

    const_value_type get_datum(size_t i) const {
        return data_.get_datum(i);
        // decompress data
        // auto result = std::vector<float>(dimensions());
        // compressed_value_type compressed = data_.get_datum(i);
        // for (size_t j = 0; j < dimensions(); ++j) {
        //     auto val = static_cast<float>(compressed[j]);
        //     result[j] = scale_ * val + bias_;
        // }

        // return result;
    }

    template <typename QueryType, size_t N>
    void set_datum(size_t i, std::span<QueryType, N> datum) {
        auto dims = dimensions();
        assert(datum.size() == dims);

        // Compression range extrema
        static constexpr std::int8_t MIN = std::numeric_limits<std::int8_t>::min();
        static constexpr std::int8_t MAX = std::numeric_limits<std::int8_t>::max();

        // Uniform scalar quantization function
        auto scalar = [&](float v) -> std::int8_t {
            return std::clamp<float>(std::round((v - bias_) / scale_), MIN, MAX);
        };

        // Prepare compressed elements
        std::vector<std::int8_t> buffer(dims);
        for (size_t j = 0; j < dims; ++j) {
            // Apply scalar quantization to element
            buffer[j] = scalar(datum[j]);
        }
        data_.set_datum(i, buffer);

        // TODO: Float16 truncation check? (see codec.h, line 114)
    }

    template <data::ImmutableMemoryDataset Dataset>
    static SQDataset compress(const Dataset& data, const allocator_type& allocator = {}) {
        return compress(data, 1, allocator);
    }

    template <data::ImmutableMemoryDataset Dataset>
    static SQDataset compress(
        const Dataset& data, size_t num_threads, const allocator_type& allocator = {}
    ) {
        auto pool = threads::DefaultThreadPool{num_threads};
        return compress(data, pool, allocator);
    }

    template <data::ImmutableMemoryDataset Dataset, threads::ThreadPool Pool>
    static SQDataset
    compress(const Dataset& data, Pool& threadpool, const allocator_type& allocator = {}) {
        if (Extent != Dynamic && data.dimensions() != Extent) {
            throw ANNEXCEPTION("Dimension mismatch!");
        }

        static constexpr size_t batch_size = 512;

        // Helper struct to collect values
        struct Accumulator {
            double min = 0.0;
            double max = 0.0;

            void accumulate(double val) {
                min = std::min(min, val);
                max = std::max(max, val);
            }

            void merge(const Accumulator& other) {
                min = std::min(min, other.min);
                max = std::max(max, other.max);
            }
        };

        // Thread-local accumulators
        std::vector<Accumulator> tls(threadpool.size());

        // Compute mean and squared sum
        threads::parallel_for(
            threadpool,
            threads::DynamicPartition(data.size(), batch_size),
            [&](const auto& indices, uint64_t tid) {
                threads::UnitRange range{indices};
                Accumulator local;

                for (size_t i = range.start(); i < range.stop(); ++i) {
                    const auto& datum = data.get_datum(i);
                    for (size_t d = 0; d < data.dimensions(); ++d) {
                        local.accumulate(datum[d]);
                    }
                }

                tls.at(tid).merge(local);
            }
        );

        // Reduce
        Accumulator global;
        for (const auto& partial : tls) {
            global.merge(partial);
        }

        // Compress the scaled and biased values
        // TODO: Templated compression bits
        // static constexpr size_t bits = 8;

        // Compression range extrema
        static constexpr std::int8_t MIN = std::numeric_limits<std::int8_t>::min();
        static constexpr std::int8_t MAX = std::numeric_limits<std::int8_t>::max();

        // Compute scale and bias
        float scale = (global.max - global.min) / (MAX - MIN);
        float bias = global.min - MIN * scale;

        // Uniform scalar quantization function
        auto scalar = [&](float v) -> std::int8_t {
            return std::clamp<float>(std::round((v - bias) / scale), MIN, MAX);
        };

        data_type compressed{data.size(), data.dimensions(), allocator};

        threads::parallel_for(
            threadpool,
            threads::DynamicPartition(data.size(), batch_size),
            [&](const auto& indices, uint64_t /*tid*/) {
                threads::UnitRange range{indices};
                for (size_t i = range.start(); i < range.stop(); ++i) {
                    // Load original row
                    auto original = data.get_datum(i);

                    // Allocate temporary buffer for transformed data
                    std::vector<std::int8_t> transformed(original.size());

                    for (size_t d = 0; d < original.size(); ++d) {
                        float val = static_cast<float>(original[d]);
                        transformed[d] = scalar(val);
                    }

                    // Store normalized data back (set_datum will do narrowing if needed)
                    compressed.set_datum(i, transformed);
                }
            }
        );

        return SQDataset<Extent, Alloc>{std::move(compressed), scale, bias};
    }

    /// @brief Save dataset to a file.
    static constexpr lib::Version save_version = scalar_quantization_save_version;
    static constexpr std::string_view serialization_schema =
        scalar_quantization_serialization_schema;
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(data, ctx),
             {"scale", lib::save(scale_, ctx)},
             {"bias", lib::save(bias_, ctx)}}
        );
    }

    /// @brief Load dataset from a file.
    static SQDataset
    load(const lib::LoadTable& table, const allocator_type& allocator = {}) {
        return SQDataset<Extent, Alloc>{
            SVS_LOAD_MEMBER_AT_(table, data, allocator),
            lib::load_at<float>(table, "scale"),
            lib::load_at<float>(table, "bias")};
    }

    /// @brief Prefetch data in the dataset.
    void prefetch(size_t i) const { data_.prefetch(i); }
};

} // namespace scalar
} // namespace quantization
} // namespace svs