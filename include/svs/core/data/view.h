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

#include "svs/concepts/data.h"
#include "svs/lib/threads/types.h"

// stl
#include <functional>
#include <span>
#include <type_traits>
#include <vector>

namespace svs::data {

namespace detail {

template <typename I>
void check_indices(const threads::UnitRange<I>& indices, size_t size) {
    if (indices.front() < 0 || lib::narrow<size_t>(indices.back()) >= size) {
        throw ANNEXCEPTION(
            "Invalid indices [{}, {}) for a range of size {}.",
            indices.front(),
            indices.back() + 1,
            size
        );
    }
}

template <std::integral I, size_t N>
void check_indices(std::span<const I, N> indices, size_t size) {
    for (auto i : indices) {
        bool okay = true;
        if constexpr (std::is_signed_v<I>) {
            okay &= (i >= 0);
        }
        okay &= (i < size);
        if (!okay) {
            throw ANNEXCEPTION("Trying to index range [0, {}) with an index {}", size, i);
        }
    }
}

template <std::integral I, typename Alloc>
void check_indices(const std::vector<I, Alloc>& indices, size_t size) {
    check_indices(lib::as_const_span(indices), size);
}

template <typename Data, typename Indices> class DataViewImpl {
  public:
    using raw_data_type = std::remove_const_t<Data>;
    using raw_reference = Data&;
    using raw_const_reference = const Data&;

    using value_type = typename raw_data_type::value_type;
    using const_value_type = typename raw_data_type::const_value_type;

    ///// Constructors
    DataViewImpl(raw_reference data, const Indices& indices)
        : data_{data}
        , indices_{indices} {
        check_indices(indices_, data.size());
    }

    DataViewImpl(raw_reference data, Indices&& indices)
        : data_{data}
        , indices_{std::move(indices)} {
        check_indices(indices_, data.size());
    }

    ///// Data interface
    size_t size() const { return indices_.size(); }
    size_t dimensions() const { return data_.get().dimensions(); }

    const_value_type get_datum(size_t i) const {
        return data_.get().get_datum(parent_id(i));
    }

    void prefetch(size_t i) const { data_.get().prefetch(parent_id(i)); }

    template <typename T>
    void set_datum(size_t i, const T& v)
        requires MemoryDataset<raw_data_type>
    {
        data_.get().set_datum(parent_id(i), v);
    }

    ///
    /// @brief Return the parent ID for index ``i``.
    ///
    size_t parent_id(size_t i) const {
        assert(i <= size());
        return indices_[i];
    }

    /// @brief Return the parent dataset.
    raw_const_reference& parent() const { return data_; }

    /// @brief Return the parent dataset.
    raw_reference& parent() { return data_; }

    /// @brief Return the container of the parent indices.
    const Indices& parent_indices() const { return indices_; }

    /// @brief Return an iterator yielding each valid index of the view.
    threads::UnitRange<size_t> eachindex() const {
        return threads::UnitRange<size_t>(0, size());
    }

  private:
    std::reference_wrapper<Data> data_;
    Indices indices_;
};

} // namespace detail

template <ImmutableMemoryDataset Data, typename Indices>
class ConstDataView : public detail::DataViewImpl<const Data, Indices> {
  private:
    using parent_type = detail::DataViewImpl<const Data, Indices>;

  public:
    ///// Constructors
    ConstDataView(const Data& data, const Indices& indices)
        : parent_type{data, indices} {}
    ConstDataView(const Data& data, Indices&& indices)
        : parent_type{data, std::move(indices)} {}
};

template <MemoryDataset Data, typename Indices>
class DataView : public detail::DataViewImpl<Data, Indices> {
  private:
    using parent_type = detail::DataViewImpl<Data, Indices>;

  public:
    ///// Constructors
    DataView(Data& data, const Indices& indices)
        : parent_type{data, indices} {}
    DataView(Data& data, Indices&& indices)
        : parent_type{data, std::move(indices)} {}
};

///// Entry Points

template <MemoryDataset Data, typename Indices>
DataView<Data, std::decay_t<Indices>> make_view(Data& data, Indices&& indices) {
    return DataView{data, std::forward<Indices>(indices)};
}

template <ImmutableMemoryDataset Data, typename Indices>
ConstDataView<Data, std::decay_t<Indices>>
make_const_view(const Data& data, Indices&& indices) {
    return ConstDataView{data, std::forward<Indices>(indices)};
}

template <ImmutableMemoryDataset Data, typename Indices>
ConstDataView<Data, std::decay_t<Indices>> make_view(const Data& data, Indices&& indices) {
    return make_const_view(data, std::forward<Indices>(indices));
}

} // namespace svs::data
