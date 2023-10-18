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

#include "svs/concepts/data.h"
#include "svs/lib/threads/types.h"

// stl
#include <functional>
#include <type_traits>

namespace svs::data {

namespace detail {

template <typename Data, typename I>
void check_indices(const Data& data, const threads::UnitRange<I>& indices) {
    if (indices.front() < 0 || lib::narrow<size_t>(indices.back()) > data.size()) {
        throw ANNEXCEPTION("Invalid indices");
    }
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
        check_indices(data, indices_);
    }

    DataViewImpl(raw_reference data, Indices&& indices)
        : data_{data}
        , indices_{std::move(indices)} {
        check_indices(data, indices_);
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
