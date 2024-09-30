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
// header under test
#include "svs/lib/invoke.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace {

namespace ns {

struct AddOverload {
    template <typename T>
    svs::svs_invoke_result_t<AddOverload, const T&, int> operator()(const T& x) const {
        return svs::svs_invoke(*this, x, 10);
    }
};

inline constexpr AddOverload custom_add = {};

// Default implementation
int svs_invoke(svs::tag_t<custom_add>, int x, int y) { return x + y; }

} // namespace ns

namespace ns_a {

struct A {
  public:
    int value_;
};

double svs_invoke(svs::tag_t<ns::custom_add>, A a, int y) {
    return static_cast<double>(2 * a.value_ + y);
}

} // namespace ns_a

} // namespace

CATCH_TEST_CASE("Invoke", "[lib][svs_invoke]") {
    static_assert(std::is_same_v<svs::tag_t<ns::custom_add>, ns::AddOverload>);
    static_assert(svs::svs_invocable<ns::AddOverload, int, int>);
    static_assert(svs::svs_invocable<ns::AddOverload, ns_a::A, int>);
    static_assert(!svs::svs_invocable<ns::AddOverload, std::string, int>);

    static_assert(std::is_same_v<svs::svs_invoke_result_t<ns::AddOverload, int, int>, int>);
    static_assert(std::is_same_v<
                  svs::svs_invoke_result_t<ns::AddOverload, ns_a::A, int>,
                  double>);

    CATCH_REQUIRE(ns::custom_add(10) == 20);
    CATCH_REQUIRE(ns::custom_add(ns_a::A{20}) == 2 * 20 + 10);
}
