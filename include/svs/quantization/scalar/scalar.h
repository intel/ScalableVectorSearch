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

// svs
#include "svs/core/data/simple.h"
#include "svs/core/distance.h"
#include "svs/lib/memory.h"
#include "svs/lib/static.h"
#include "svs/lib/version.h"

// stl
#include <concepts>
#include <limits>
#include <memory>
#include <variant>

namespace svs {
namespace quantization {
namespace scalar {

namespace detail {
template <typename Original, typename Compressed>
Compressed compress(Original val, float scale, float bias) {
    static constexpr auto MIN = std::numeric_limits<Compressed>::min();
    static constexpr auto MAX = std::numeric_limits<Compressed>::max();
    return std::clamp<float>(std::round((val - bias) / scale), MIN, MAX);
}

template <typename Compressed> float decompress(Compressed val, float scale, float bias) {
    return scale * float(val) + bias;
}

// Used to SFINAE away resizing methods if the allocator is not blocked.
template <typename A> inline constexpr bool is_blocked = false;
template <typename A> inline constexpr bool is_blocked<data::Blocked<A>> = true;

} // namespace detail

// Trait to determine if an allocator is blocked or not.
template <typename A>
concept is_resizeable = detail::is_blocked<A>;

template <typename ElementType> class EuclideanCompressed {
  public:
    using compare = std::less<>;

    static constexpr bool implicit_broadcast = false;
    static constexpr bool must_fix_argument = true;

    EuclideanCompressed(float scale, float bias, size_t dims)
        : query_compressed_{1, dims}
        , scale_{scale}
        , bias_{bias}
        , scale_sq_{scale * scale} {}

    EuclideanCompressed shallow_copy() const {
        return EuclideanCompressed(scale_, bias_, query_compressed_.dimensions());
    }

    template <typename T> void fix_argument(const std::span<T>& query) {
        // Store the compressed query
        std::vector<ElementType> compressed(query.size());
        std::transform(query.begin(), query.end(), compressed.begin(), [&](T v) {
            return detail::compress<T, ElementType>(v, scale_, bias_);
        });
        query_compressed_.set_datum(0, compressed);
    }

    std::span<const ElementType> view_query() const {
        return query_compressed_.get_datum(0);
    }

    template <typename T>
    float compute(const T& y) const
        requires(lib::is_spanlike_v<T>)
    {
        auto inner = distance::DistanceL2{};
        return scale_sq_ * distance::compute(inner, view_query(), y);
    }

    float get_scale() const { return scale_; }
    float get_bias() const { return bias_; }

  private:
    data::SimpleData<ElementType> query_compressed_;
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
        , bias_{bias} {}

    InnerProductCompressed shallow_copy() const {
        return InnerProductCompressed(scale_, bias_, query_fp32_.dimensions());
    }

    template <typename T> void fix_argument(const std::span<T>& query) {
        query_fp32_.set_datum(0, query);
        const auto query_fp32 = query_fp32_.get_datum(0);
        auto sum =
            std::reduce(query_fp32.begin(), query_fp32.end(), 0.0F, [](float acc, float v) {
                return acc + float(v);
            });
        offset_ = bias_ * sum;
    }

    std::span<const float> view_query() const { return query_fp32_.get_datum(0); }

    template <typename T>
    float compute(const T& y) const
        requires(lib::is_spanlike_v<T>)
    {
        auto inner = distance::DistanceIP{};
        float ip = distance::compute(inner, view_query(), y);
        return scale_ * ip + offset_;
    }

    float get_scale() const { return scale_; }
    float get_bias() const { return bias_; }

  private:
    data::SimpleData<float> query_fp32_;
    float scale_;
    float bias_;

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
        , bias_{bias}
        , inner_{} {}

    template <typename T> void fix_argument(const std::span<T>& query) {
        query_fp32_.set_datum(0, query);
        inner_.fix_argument(query);
    }

    std::span<const float> view_query() const { return query_fp32_.get_datum(0); }

    template <typename T>
    float compute(const T& y) const
        requires(lib::is_spanlike_v<T>)
    {
        std::vector<float> y_decomp(y.size());
        std::transform(y.begin(), y.end(), y_decomp.begin(), [&](auto v) {
            return detail::decompress<float>(v, scale_, bias_);
        });
        return distance::compute(inner_, view_query(), std::span<const float>(y_decomp));
    }

    float get_scale() const { return scale_; }
    float get_bias() const { return bias_; }

  private:
    data::SimpleData<float> query_fp32_;
    float scale_;
    float bias_;

    distance::DistanceCosineSimilarity inner_;
};

namespace detail {

struct MinMaxAccumulator {
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();

    void accumulate(float val) {
        min = std::min(min, val);
        max = std::max(max, val);
    }

    void merge(const MinMaxAccumulator& other) {
        min = std::min(min, other.min);
        max = std::max(max, other.max);
    }
};

// Operator to find global min and max in dataset
struct MinMax {
    template <data::ImmutableMemoryDataset Dataset, threads::ThreadPool Pool>
    MinMaxAccumulator operator()(const Dataset& data, Pool& threadpool) {
        // Thread-local accumulators
        std::vector<MinMaxAccumulator> tls(threadpool.size());

        // Compute min and max values in dataset
        threads::parallel_for(
            threadpool,
            threads::StaticPartition{data.size()},
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

// Operator to compress a dataset using a threadpool
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
        data_type compressed{data.size(), data.dimensions(), allocator};

        threads::parallel_for(
            threadpool,
            threads::StaticPartition{data.size()},
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
template <typename T, typename ElementType> struct CompressedDistance;

template <typename ElementType>
struct CompressedDistance<distance::DistanceL2, ElementType> {
    using type = EuclideanCompressed<ElementType>;
};

template <typename ElementType>
struct CompressedDistance<distance::DistanceIP, ElementType> {
    using type = InnerProductCompressed;
};

template <typename ElementType>
struct CompressedDistance<distance::DistanceCosineSimilarity, ElementType> {
    using type = CosineSimilarityCompressed;
};

// Trait to identify whether a type has `uses_compressed_data`
template <typename T, typename = void> struct compressed_data_trait : std::false_type {};

// Specialization for types that have `uses_compressed_data == true`
template <typename T>
struct compressed_data_trait<T, std::void_t<decltype(T::uses_compressed_data)>>
    : std::bool_constant<T::uses_compressed_data> {};

template <typename T>
inline constexpr bool compressed_data_trait_v = compressed_data_trait<T>::value;

} // namespace detail

template <typename Distance, typename ElementType>
using compressed_distance_t =
    typename detail::CompressedDistance<Distance, ElementType>::type;

template <typename T>
concept IsSQData = detail::compressed_data_trait_v<T>;

class Decompressor {
  public:
    Decompressor() = delete;
    Decompressor(float scale, float bias)
        : scale_(scale)
        , bias_(bias) {}

    template <typename T> std::span<const float> operator()(const T& y) {
        std::transform(y.begin(), y.end(), buffer_.begin(), [&](auto v) {
            return detail::decompress<float>(v, scale_, bias_);
        });
        return lib::as_const_span(buffer_);
    }

  private:
    float scale_ = {};
    float bias_ = {};
    std::vector<float> buffer_ = {};
};

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

    ///// Decompressor
    Decompressor decompressor() const { return Decompressor{scale_, bias_}; }

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
        float scale = (global.max - global.min) / (MAX - MIN);
        float bias = global.min - MIN * scale;

        // Compress data
        auto compressor = detail::Compressor<element_type, data_type>{scale, bias};
        auto compressed = compressor(data, threadpool, allocator);

        return SQDataset<element_type, extent, allocator_type>{
            std::move(compressed), scale, bias};
    }

    /// @brief Compact the dataset
    template <std::integral I, threads::ThreadPool Pool>
    void compact(std::span<const I> new_to_old, Pool& threadpool, size_t batchsize)
        requires is_resizeable<Alloc>
    {
        data_.compact(new_to_old, threadpool, batchsize);
    }

    /// @brief Resize the dataset
    void resize(size_t new_size)
        requires is_resizeable<Alloc>
    {
        data_.resize(new_size);
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

/////
///// Support for index building.
/////

///
/// Adaptor to adjust a distance function with type `Distance` to enable index building
/// over a compressed dataset.
///
/// Essentially, allows for distance computations between two elements of a compressed
/// dataset.
///
template <typename Distance> class DecompressionAdaptor {
  public:
    using distance_type = Distance;
    using compare = distance::compare_t<distance_type>;
    static constexpr bool implicit_broadcast = false;
    static constexpr bool must_fix_argument = true;

    DecompressionAdaptor(const distance_type& inner, size_t size_hint = 0)
        : inner_{inner}
        , decompressed_(size_hint) {}

    DecompressionAdaptor(distance_type&& inner, size_t size_hint = 0)
        : inner_{std::move(inner)}
        , decompressed_(size_hint) {}

    ///
    /// @brief Construct the internal portion of DecompressionAdaptor directly.
    ///
    /// The goal of the decompression adaptor is to wrap around an inner distance functor
    /// and decompress the left-hand component when requested, forwarding the decompressed
    /// value to the inner functor upon future distance computations.
    ///
    /// The inner distance functor may have non-trivial state associated with it.
    /// This constructor allows to construction of that inner functor directly to avoid
    /// a copy or move constructor.
    ///
    template <typename... Args>
    DecompressionAdaptor(std::in_place_t SVS_UNUSED(tag), Args&&... args)
        : inner_{std::forward<Args>(args)...}
        , decompressed_() {}

    DecompressionAdaptor shallow_copy() const {
        return DecompressionAdaptor(inner_, decompressed_.size());
    }

    // Distance API.
    template <typename Left> void fix_argument(Left left) {
        decompressed_.resize(left.size());
        std::transform(left.begin(), left.end(), decompressed_.begin(), [&](auto v) {
            return detail::decompress<float>(v, inner_.get_scale(), inner_.get_bias());
        });
        inner_.fix_argument(view());
    }

    template <typename Right> float compute(const Right& right) const {
        return distance::compute(inner_, view(), right);
    }

    std::span<const float> view() const { return decompressed_; }

  private:
    distance_type inner_;
    std::vector<float> decompressed_;
};

/////
///// Decompression Accessor
/////

// A composition of ``GetDatumAccessor`` and a vector decompressor.
class DecompressionAccessor {
  public:
    template <IsSQData Data>
    DecompressionAccessor(const Data& data)
        : decompressor_{data.get_scale(), data.get_bias()} {}

    // Access
    template <IsSQData Data> std::span<const float> operator()(const Data& data, size_t i) {
        return decompressor_(data.get_datum(i));
    }

  private:
    scalar::Decompressor decompressor_;
};

template <IsSQData Data, typename Distance>
DecompressionAdaptor<compressed_distance_t<Distance, typename Data::element_type>>
adapt_for_self(const Data& data, const Distance& SVS_UNUSED(distance)) {
    return DecompressionAdaptor<
        compressed_distance_t<Distance, typename Data::element_type>>(
        std::in_place, data.get_scale(), data.get_bias(), data.dimensions()
    );
}

} // namespace scalar
} // namespace quantization
} // namespace svs
