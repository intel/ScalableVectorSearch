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
        , scale_T_{1.0F / scale} {}

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
        return scale_T_ * distance::compute(inner, view_query(), y);
    }

  private:
    data::SimpleData<float> query_fp32_;
    float scale_;
    float bias_;

    float scale_T_;
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
        , scale_sq_T_{1.0F / (scale * scale)} {}

    InnerProductCompressed shallow_copy() const {
        return InnerProductCompressed(scale_, bias_, query_fp32_.dimensions());
    }

    template <typename T> void fix_argument(const std::span<T>& query) {
        query_fp32_.set_datum(0, query);
    }

    std::span<const float> view_query() const { return query_fp32_.get_datum(0); }

    template <typename T>
    float compute(const T& y) const
        requires(lib::is_spanlike_v<T>)
    {
        auto inner = distance::DistanceIP{};
        float sum = eve::algo::reduce(y, 0.0f);
        float ip = distance::compute(inner, view_query(), y);
        return scale_sq_T_ * (ip - bias_ * sum);
    }

  private:
    data::SimpleData<float> query_fp32_;
    float scale_;
    float bias_;

    // pre-computed values
    float scale_sq_T_;
};

class CosineSimilarityCompressed {
  public:
    using compare = std::greater<>;

    static constexpr bool implicit_broadcast = false;
    static constexpr bool must_fix_argument = true;

    CosineSimilarityCompressed(float scale, float bias, size_t dims)
        : query_fp32_{1, dims}
        , scale_{scale}
        , bias_{bias}
        , inner_{}
        , scale_T_{1.0F / scale} {}

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
            return (v - bias_) * scale_T_;
        });
        return distance::compute(inner_, view_query(), std::span<const float>(y_biased));
    }

  private:
    data::SimpleData<float> query_fp32_;
    float scale_;
    float bias_;

    distance::DistanceCosineSimilarity inner_;
    float scale_T_;
};

namespace detail {

template <typename Original, typename Compressed>
Compressed compress(Original val, float scale, float bias) {
    static constexpr auto MIN = std::numeric_limits<Compressed>::min();
    static constexpr auto MAX = std::numeric_limits<Compressed>::max();
    return std::clamp<float>(std::round(scale * val + bias), MIN, MAX);
}

template <typename Compressed> float decompress(Compressed val, float scale, float bias) {
    return (val - bias) / scale;
}

struct MinMaxAccumulator {
    double min = 0.0;
    double max = 0.0;

    void accumulate(double val) {
        min = std::min(min, val);
        max = std::max(max, val);
    }

    void merge(const MinMaxAccumulator& other) {
        min = std::min(min, other.min);
        max = std::max(max, other.max);
    }
};

// operator to find global min and max in dataset
struct MinMax {
    template <data::ImmutableMemoryDataset Dataset, threads::ThreadPool Pool>
    MinMaxAccumulator operator()(const Dataset& data, Pool& threadpool) {
        static constexpr size_t batch_size = 512;

        // Thread-local accumulators
        std::vector<MinMaxAccumulator> tls(threadpool.size());

        // Compute mean and squared sum
        threads::parallel_for(
            threadpool,
            threads::DynamicPartition(data.size(), batch_size),
            [&](const auto& indices, uint64_t tid) {
                threads::UnitRange range{indices};
                MinMaxAccumulator local;

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
        MinMaxAccumulator global;
        for (const auto& partial : tls) {
            global.merge(partial);
        }

        return global;
    }
};

// operator to compress a dataset using a threadpool
template <typename Element, typename Data> struct Compressor {
    using element_type = Element;
    using data_type = Data;

    Compressor(float scale, float bias)
        : scale_{scale}
        , bias_{bias} {}

    template <
        data::ImmutableMemoryDataset Dataset,
        threads::ThreadPool Pool,
        typename Alloc>
    data_type
    operator()(const Dataset& data, Pool& threadpool, const Alloc& allocator) const {
        static constexpr size_t batch_size = 512;

        data_type compressed{data.size(), data.dimensions(), allocator};

        threads::parallel_for(
            threadpool,
            threads::DynamicPartition(data.size(), batch_size),
            [&](const auto& indices, uint64_t /*tid*/) {
                threads::UnitRange range{indices};
                // Allocate a buffer of given dimensionality, will be re-used for each datum
                std::vector<element_type> buffer(data.dimensions());
                for (size_t i = range.start(); i < range.stop(); ++i) {
                    // Compress datum
                    auto datum = data.get_datum(i);
                    std::transform(
                        datum.begin(),
                        datum.end(),
                        buffer.begin(),
                        [&](float v) {
                            return compress<float, element_type>(v, scale_, bias_);
                        }
                    );
                    // Store to compressed dataset
                    compressed.set_datum(i, buffer);
                }
            }
        );

        return compressed;
    }

  private:
    float scale_;
    float bias_;
};

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
template <typename T, size_t Extent = svs::Dynamic, typename Alloc = lib::Allocator<T>>
class SQDataset {
  public:
    constexpr static size_t extent = Extent;
    constexpr static bool uses_compressed_data = true;

    using allocator_type = Alloc;
    using element_type = T;
    using data_type = data::SimpleData<element_type, Extent, allocator_type>;
    using const_value_type = std::span<const element_type, Extent>;
    using value_type = const_value_type;

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

    const_value_type get_datum(size_t i) const { return data_.get_datum(i); }

    std::vector<float> decompress_datum(size_t i) const {
        auto datum = get_datum(i);
        std::vector<float> buffer(datum.size());
        std::transform(datum.begin(), datum.end(), buffer.begin(), [&](element_type v) {
            return detail::decompress(v, scale_, bias_);
        });
        return buffer;
    }

    template <typename QueryType, size_t N>
    void set_datum(size_t i, std::span<QueryType, N> datum) {
        auto dims = dimensions();
        assert(datum.size() == dims);

        // Compress elements
        std::vector<element_type> buffer(dims);
        std::transform(datum.begin(), datum.end(), buffer.begin(), [&](QueryType v) {
            return detail::compress<QueryType, element_type>(v, scale_, bias_);
        });

        data_.set_datum(i, buffer);
        // TODO: Float16 truncation check? (see codec.h, line 1[14)
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

        // Get dataset extrema
        auto minmax = detail::MinMax{};
        auto global = minmax(data, threadpool);

        // Compute scale and bias
        constexpr float MIN = std::numeric_limits<element_type>::min();
        constexpr float MAX = std::numeric_limits<element_type>::max();
        float scale = (MAX - MIN) / (global.max - global.min);
        float bias = MIN - scale * global.min;

        // Compress data
        auto compressor = detail::Compressor<element_type, data_type>{scale, bias};
        auto compressed = compressor(data, threadpool, allocator);

        return SQDataset<element_type, extent, allocator_type>{
            std::move(compressed), scale, bias};
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
        return SQDataset<element_type, extent, allocator_type>{
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