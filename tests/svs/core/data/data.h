/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
 */
// svs
#include "svs/core/data.h"

// stl
#include <vector>

namespace svs_test::data {

// A mock dataset with integer entries.
class MockDataset {
  public:
    using value_type = int64_t;
    using const_value_type = int64_t;

    template <typename AccessType> using mode_value_type = value_type;
    template <typename AccessType> using mode_const_value_type = const_value_type;

    struct Iota {
        size_t start;
        size_t step;
        size_t count;
    };

    // Constructor
    MockDataset(Iota iota)
        : data_{} {
        for (size_t i = 0; i < iota.count; ++i) {
            data_.push_back(iota.start + iota.step * i);
        }
    }

    // Dataset API
    size_t size() const { return data_.size(); }
    size_t dimensions() const { return 1; }
    const_value_type get_datum(size_t i) const { return data_.at(i); }
    void prefetch(size_t) const {}
    void set_datum(size_t i, value_type val) { data_.at(i) = val; }

  private:
    std::vector<int64_t> data_;
};

template <typename T> void set_sequential(T& x) {
    size_t count = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        for (auto& j : x.get_datum(i)) {
            j = count;
            ++count;
        }
    }
}

template <typename T> bool is_sequential(const T& x) {
    size_t count = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        for (auto& j : x.get_datum(i)) {
            if (j != count) {
                return false;
            }
            ++count;
        }
    }
    return true;
}
} // namespace svs_test::data
