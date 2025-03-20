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
#include "svs/lib/memory.h"
#include "svs/lib/static.h"
#include "svs/lib/version.h"

// stl
#include <memory>

namespace svs {
namespace quantization {
namespace scalar {

inline constexpr std::string_view scalar_quantization_serialization_schema =
    "scalar_quantization_dataset";
inline constexpr lib::Version scalar_quantization_save_version = lib::Version(0, 0, 0);

// Scalar Quantization Dataset
// This class provides a globally quantized (scale & bias) dataset.
template <size_t Extent = svs::Dynamic, typename Alloc = lib::Allocator<std::int8_t>>
class SQDataset {
  public:
    constexpr static size_t extent = Extent;

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
    using compressed_value_type = std::span<const std::int8_t, Extent>;
    using const_value_type = std::vector<float>;
    using value_type = const_value_type;

  private:
    float scale_;
    float bias_;
    data_type data_;

  public:
    SQDataset(size_t size, size_t dims);
    SQDataset(data_type data, float scale, float bias);

    size_t size() const;
    size_t dimensions() const;

    float get_scale() const { return scale_; }
    float get_bias() const { return bias_; }

    const_value_type get_datum(size_t i) const;

    template <typename QueryType, size_t N>
    void set_datum(size_t i, std::span<QueryType, N> datum);

    template <data::ImmutableMemoryDataset Dataset>
    static SQDataset compress(const Dataset& data, const allocator_type& allocator = {});

    template <data::ImmutableMemoryDataset Dataset>
    static SQDataset
    compress(const Dataset& data, size_t num_threads, const allocator_type& allocator = {});

    template <data::ImmutableMemoryDataset Dataset, threads::ThreadPool Pool>
    static SQDataset
    compress(const Dataset& data, Pool& threadpool, const allocator_type& allocator = {});

    static constexpr lib::Version save_version = scalar_quantization_save_version;
    static constexpr std::string_view serialization_schema =
        scalar_quantization_serialization_schema;
    lib::SaveTable save(const lib::SaveContext& ctx) const;

    static SQDataset
    load(const lib::LoadTable& table, const allocator_type& allocator = {});
};

} // namespace scalar
} // namespace quantization
} // namespace svs