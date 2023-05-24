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

// header under test
#include "svs/core/medioid.h"

// svs
#include "svs/lib/timing.h"

// test utils
#include "tests/utils/test_dataset.h"

// catch2
#include "catch2/catch_test_macros.hpp"

///// helpers
namespace {
template <typename Data, typename Pred = svs::lib::ReturnsTrueType>
std::vector<double>
medioid_reference(const Data& data, const Pred& predicate = svs::lib::ReturnsTrueType()) {
    std::vector<double> sums(data.dimensions(), 0.0);
    size_t count = 0;
    for (size_t i = 0, upper = data.size(); i < upper; ++i) {
        if (!predicate(i)) {
            continue;
        }

        ++count;
        const auto& datum = data.get_datum(i);
        for (size_t j = 0, jupper = data.dimensions(); j < jupper; ++j) {
            sums[j] += datum[j];
        }
    }

    CATCH_REQUIRE(count > 0);
    for (size_t j = 0, jupper = sums.size(); j < jupper; ++j) {
        sums[j] /= count;
    }
    return sums;
}

template <typename Data, typename Pred = svs::lib::ReturnsTrueType>
std::vector<double>
medioid_countsum(const Data& data, const Pred& predicate = svs::lib::ReturnsTrueType()) {
    svs::utils::CountSum sums(data.dimensions());
    for (size_t i = 0, upper = data.size(); i < upper; ++i) {
        if (!predicate(i)) {
            continue;
        }
        sums.add(data.get_datum(i));
    }
    return sums.finish();
}

template <typename Data, typename Pred = svs::lib::ReturnsTrueType>
std::vector<double>
compute_variances(const Data& data, const Pred& predicate = svs::lib::ReturnsTrueType()) {
    std::vector<double> means = medioid_reference(data, predicate);
    std::vector<double> variances(means.size(), 0.0);

    size_t count{0};
    for (size_t i = 0, upper = data.size(); i < upper; ++i) {
        if (!predicate(i)) {
            continue;
        }

        const auto& datum = data.get_datum(i);
        ++count;
        for (size_t j = 0, jupper = data.dimensions(); j < jupper; ++j) {
            double temp = static_cast<double>(datum[j]) - static_cast<double>(means[j]);
            variances[j] += temp * temp;
        }
    }

    CATCH_REQUIRE(count > 0);
    for (size_t j = 0, jupper = variances.size(); j < jupper; ++j) {
        variances[j] /= count;
    }
    return variances;
}

} // namespace

CATCH_TEST_CASE("Testing Medioid Computation", "[core][medioid]") {
    auto data = test_dataset::data_f32();
    auto only_odds = [](const size_t i) { return (i % 2) != 0; };
    auto returns_true = svs::lib::ReturnsTrueType();
    svs::utils::PairwiseSumParameters test_parameters{100, 1000};

    CATCH_SECTION("CountSum Data Structure") {
        // No predicate
        std::vector<double> ref = medioid_reference(data);
        std::vector<double> from_countsum = medioid_countsum(data);
        auto identity = svs::lib::identity();
        svs::utils::CountSum countsum = svs::utils::op_pairwise(
            data,
            svs::utils::CountSum(data.dimensions()),
            svs::threads::UnitRange<size_t>(0, data.size()),
            returns_true,
            identity,
            test_parameters
        );

        CATCH_REQUIRE(countsum.count == data.size());
        std::vector<double> from_pairwise = countsum.finish();

        CATCH_REQUIRE(ref.size() == from_countsum.size());
        CATCH_REQUIRE(std::equal(ref.begin(), ref.end(), from_countsum.begin()));

        CATCH_REQUIRE(ref.size() == from_pairwise.size());
        CATCH_REQUIRE(std::equal(ref.begin(), ref.end(), from_pairwise.begin()));

        // With Predicate
        ref = medioid_reference(data, only_odds);
        from_countsum = medioid_countsum(data, only_odds);
        countsum = svs::utils::op_pairwise(
            data,
            svs::utils::CountSum(data.dimensions()),
            svs::threads::UnitRange<size_t>(0, data.size()),
            only_odds,
            identity,
            test_parameters
        );
        CATCH_REQUIRE(ref.size() == from_countsum.size());
        CATCH_REQUIRE(std::equal(ref.begin(), ref.end(), from_countsum.begin()));

        CATCH_REQUIRE(countsum.count == data.size() / 2);
        from_pairwise = countsum.finish();
        CATCH_REQUIRE(ref.size() == from_pairwise.size());
        CATCH_REQUIRE(std::equal(ref.begin(), ref.end(), from_pairwise.begin()));
    }

    CATCH_SECTION("Parallelized") {
        svs::threads::NativeThreadPool threadpool(2);

        // No predicate
        auto tic = svs::lib::now();
        std::vector<double> ref = medioid_reference(data);
        auto diff = svs::lib::time_difference(tic);
        std::cout << "Medioid reference time: " << diff << std::endl;

        tic = svs::lib::now();
        std::vector<double> test = svs::utils::compute_medioid(
            data,
            threadpool,
            svs::lib::ReturnsTrueType(),
            svs::lib::identity(),
            test_parameters
        );
        diff = svs::lib::time_difference(tic);
        std::cout << "Medioid parallel time: " << diff << std::endl;

        CATCH_REQUIRE(ref.size() == test.size());
        CATCH_REQUIRE(std::equal(ref.begin(), ref.end(), test.begin()));

        // With predicate
        ref = medioid_reference(data, only_odds);
        test = svs::utils::compute_medioid(
            data, threadpool, only_odds, svs::lib::identity(), test_parameters
        );
        CATCH_REQUIRE(ref.size() == test.size());
        CATCH_REQUIRE(std::equal(ref.begin(), ref.end(), test.begin()));
    }

    CATCH_SECTION("Find Medioid") {
        svs::threads::NativeThreadPool threadpool(2);
        size_t index = svs::utils::find_medioid(
            data, threadpool, returns_true, svs::lib::identity(), test_parameters
        );

        std::ifstream stream(test_dataset::metadata_file(), std::ios::binary);
        auto precomputed_index = svs::lib::read_binary<uint32_t>(stream);
        CATCH_REQUIRE(index == precomputed_index);
    }

    CATCH_SECTION("Variances") {
        // Reference computation.
        std::vector<double> ref = compute_variances(data);

        // Parallel Computation
        svs::threads::NativeThreadPool threadpool(2);
        auto means = svs::utils::compute_medioid(data, threadpool);
        auto variances =
            svs::utils::op_pairwise(data, svs::utils::CountVariance(means), threadpool);

        CATCH_REQUIRE(ref.size() == variances.size());
        CATCH_REQUIRE(std::equal(
            ref.begin(),
            ref.end(),
            variances.begin(),
            [](const auto& left, const auto& right) {
                return left == Catch::Approx(right).epsilon(0.00001);
            }
        ));
    }
}
