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

// Header under test.
#include "svs/leanvec/leanvec.h"

// Extras
#include "svs/lib/saveload.h"

// test utilities
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace leanvec = svs::leanvec;
namespace lvq = svs::quantization::lvq;

namespace {

template <size_t N = svs::Dynamic> using MaybeStatic = svs::lib::MaybeStatic<N>;

template <typename T, size_t D>
[[nodiscard]] bool compare(const std::span<T, D> x, const std::span<T, D> y) {
    if (x.size() != y.size()) {
        return false;
    }
    for (size_t j = 0, jmax = x.size(); j < jmax; ++j) {
        if (x[j] != Catch::Approx(y[j]).epsilon(0.0001)) {
            return false;
        }
    }
    return true;
}

template <size_t N, size_t D, lvq::LVQPackingStrategy S>
[[nodiscard]] bool compare(
    const lvq::ScaledBiasedVector<N, D, S> x, const lvq::ScaledBiasedVector<N, D, S> y
) {
    return lvq::logically_equal(x, y);
}

template <leanvec::IsLeanDataset A, leanvec::IsLeanDataset B>
[[nodiscard]] bool compare(const A& a, const B& b) {
    if (a.size() != b.size()) {
        return false;
    }
    if (a.dimensions() != b.dimensions()) {
        return false;
    }

    for (size_t i = 0, imax = a.size(); i < imax; ++i) {
        bool eq = compare(a.get_datum(i), b.get_datum(i)) &&
                  compare(a.get_secondary(i), b.get_secondary(i));
        if (!eq) {
            return false;
        }
    }
    return true;
}

template <leanvec::IsLeanDataset T, bool HasMatcher = true>
void test_leanvec_top(bool use_pca, size_t leanvec_dims = T::leanvec_extent) {
    constexpr size_t L = T::leanvec_extent;
    constexpr size_t N = T::extent;

    // First, construct an online LeanVec.
    auto data = svs::data::SimpleData<float, N>::load(test_dataset::data_svs_file());

    if (L != svs::Dynamic) {
        CATCH_REQUIRE(leanvec_dims == L);
    }

    // To avoid needing a default-constructed dataset, use an immediately-used lambda
    // to decide the initialization path for the dataset.
    T leanvec_dataset = [&]() -> T {
        if (use_pca) {
            return T::reduce(data, 1, 0, svs::lib::MaybeStatic<L>(leanvec_dims));
        }
        return T::reduce(
            data,
            std::optional<leanvec::LeanVecMatrices<L>>(
                std::in_place,
                test_dataset::leanvec_data_matrix<L>(),
                test_dataset::leanvec_query_matrix<L>()
            ),
            1,
            0,
            svs::lib::MaybeStatic<L>(leanvec_dims)
        );
    }();

    // Try saving and reloading.
    svs_test::prepare_temp_directory();
    auto temp_dir = svs_test::temp_directory();
    svs::lib::save_to_disk(leanvec_dataset, temp_dir);
    auto reloaded_leanvec_dataset = svs::lib::load_from_disk<T>(temp_dir);
    static_assert(std::is_same_v<
                  decltype(leanvec_dataset),
                  decltype(reloaded_leanvec_dataset)>);

    // Require equality.
    CATCH_REQUIRE(compare(leanvec_dataset, reloaded_leanvec_dataset));

    // Test get_datum/set_datum
    CATCH_REQUIRE(!compare(leanvec_dataset.get_datum(10), leanvec_dataset.get_datum(100)));
    CATCH_REQUIRE(
        !compare(leanvec_dataset.get_secondary(10), leanvec_dataset.get_secondary(100))
    );

    leanvec_dataset.set_datum(100, data.get_datum(10));
    CATCH_REQUIRE(compare(leanvec_dataset.get_datum(10), leanvec_dataset.get_datum(100)));
    CATCH_REQUIRE(
        compare(leanvec_dataset.get_secondary(10), leanvec_dataset.get_secondary(100))
    );

    // Matcher
    if constexpr (HasMatcher) {
        auto m = svs::lib::load_from_disk<leanvec::Matcher>(temp_dir);
        CATCH_REQUIRE(m.primary_kind == leanvec::leanvec_kind_v<typename T::primary_data_type>);
        CATCH_REQUIRE(m.secondary_kind == leanvec::leanvec_kind_v<typename T::secondary_data_type>);
        CATCH_REQUIRE(m.leanvec_dims == leanvec_dataset.inner_dimensions());
        CATCH_REQUIRE(m.total_dims == leanvec_dataset.dimensions());

        // Invalidate the schemas of the various inner datasets.
        auto src = temp_dir / svs::lib::config_file_name;
        auto dst = temp_dir / "modified.toml";
        svs_test::mutate_table(
            src, dst, {{{"object", "primary", "__schema__"}, "invalid_schema"}}
        );
        auto ex = svs::lib::try_load_from_disk<leanvec::Matcher>(dst);
        CATCH_REQUIRE(!ex);
        CATCH_REQUIRE(ex.error() == svs::lib::TryLoadFailureReason::InvalidSchema);

        svs_test::mutate_table(
            src, dst, {{{"object", "secondary", "__schema__"}, "invalid_schema"}}
        );
        ex = svs::lib::try_load_from_disk<leanvec::Matcher>(dst);
        CATCH_REQUIRE(!ex);
        CATCH_REQUIRE(ex.error() == svs::lib::TryLoadFailureReason::InvalidSchema);

        // Modify the tables to values not supported by the matcher.
        if constexpr (leanvec::detail::is_using_lvq_tag_v<typename T::primary_data_type>) {
            svs_test::mutate_table(
                src, dst, {{{"object", "primary", "primary", "bits"}, 2}}
            );
        } else {
            svs_test::mutate_table(src, dst, {{{"object", "primary", "eltype"}, "uint8"}});
        }
        ex = svs::lib::try_load_from_disk<leanvec::Matcher>(dst);
        CATCH_REQUIRE(!ex);
        CATCH_REQUIRE(ex.error() == svs::lib::TryLoadFailureReason::Other);
    }
}

} // namespace

CATCH_TEST_CASE("LeanVec Dimensionality Reduction", "[leanvec][leandata]") {
    CATCH_SECTION("OnlineLeanVec") {
        // Make sure we can construct an instance of "OnlineLeanVec" using one of the
        // "blessed" source types.
        auto x = leanvec::OnlineLeanVec("a path!", svs::DataType::float32);
        CATCH_REQUIRE(x.path == "a path!");
        CATCH_REQUIRE(x.type == svs::DataType::float32);

        x = leanvec::OnlineLeanVec("another path!", svs::DataType::float16);
        CATCH_REQUIRE(x.path == "another path!");
        CATCH_REQUIRE(x.type == svs::DataType::float16);

        // Incompatible type.
        CATCH_REQUIRE_THROWS_AS(
            leanvec::OnlineLeanVec("another path!", svs::DataType::float64),
            svs::ANNException
        );
    }

    constexpr size_t Dyn = svs::Dynamic;
    CATCH_SECTION("LeanVec PCA without Compression") {
        CATCH_SECTION("Static") {
            test_leanvec_top<leanvec::LeanDataset<float, float, 64, 128>>(true);
            test_leanvec_top<leanvec::LeanDataset<svs::Float16, svs::Float16, 96, 128>>(true
            );
        }
        CATCH_SECTION("Dynamic") {
            test_leanvec_top<leanvec::LeanDataset<float, float, Dyn, Dyn>>(true, 64);
            test_leanvec_top<leanvec::LeanDataset<svs::Float16, svs::Float16, Dyn, Dyn>>(
                true, 96
            );
        }
    }

    CATCH_SECTION("LeanVec PCA with LVQ") {
        CATCH_SECTION("Static") {
            test_leanvec_top<
                leanvec::LeanDataset<leanvec::UsingLVQ<4>, leanvec::UsingLVQ<4>, 64, 128>>(
                true
            );
            test_leanvec_top<
                leanvec::LeanDataset<leanvec::UsingLVQ<8>, leanvec::UsingLVQ<8>, 64, 128>>(
                true
            );
            test_leanvec_top<
                leanvec::LeanDataset<leanvec::UsingLVQ<8>, svs::Float16, 64, 128>>(true);
            test_leanvec_top<
                leanvec::LeanDataset<svs::Float16, leanvec::UsingLVQ<8>, 64, 128>>(true);
        }
        CATCH_SECTION("Dynamic") {
            test_leanvec_top<
                leanvec::LeanDataset<leanvec::UsingLVQ<4>, leanvec::UsingLVQ<4>, Dyn, Dyn>>(
                true, 64
            );
            test_leanvec_top<
                leanvec::LeanDataset<leanvec::UsingLVQ<8>, leanvec::UsingLVQ<8>, Dyn, Dyn>>(
                true, 64
            );
            test_leanvec_top<
                leanvec::LeanDataset<leanvec::UsingLVQ<8>, svs::Float16, Dyn, Dyn>>(
                true, 64
            );
            test_leanvec_top<
                leanvec::LeanDataset<svs::Float16, leanvec::UsingLVQ<8>, Dyn, Dyn>>(
                true, 64
            );
        }
    }

    // Test LeanVec OOD
    CATCH_SECTION("LeanVec OOD without Compression") {
        CATCH_SECTION("Static") {
            test_leanvec_top<leanvec::LeanDataset<float, float, 64, 128>>(false);
            test_leanvec_top<leanvec::LeanDataset<svs::Float16, svs::Float16, 64, 128>>(
                false
            );
        }
        CATCH_SECTION("Dynamic") {
            test_leanvec_top<leanvec::LeanDataset<float, float, Dyn, Dyn>>(false);
            test_leanvec_top<leanvec::LeanDataset<svs::Float16, svs::Float16, Dyn, Dyn>>(
                false
            );
        }
    }

    CATCH_SECTION("LeanVec OOD with LVQ") {
        CATCH_SECTION("Static") {
            test_leanvec_top<
                leanvec::LeanDataset<leanvec::UsingLVQ<4>, leanvec::UsingLVQ<4>, 64, 128>>(
                false
            );
            test_leanvec_top<
                leanvec::LeanDataset<leanvec::UsingLVQ<8>, svs::Float16, 64, 128>>(false);
            test_leanvec_top<leanvec::LeanDataset<float, leanvec::UsingLVQ<8>, 64, 128>>(
                false
            );
        }
    }
    CATCH_SECTION("Dynamic") {
        test_leanvec_top<
            leanvec::LeanDataset<leanvec::UsingLVQ<4>, leanvec::UsingLVQ<4>, Dyn, Dyn>>(
            false, 64
        );
        test_leanvec_top<
            leanvec::LeanDataset<leanvec::UsingLVQ<8>, svs::Float16, Dyn, Dyn>>(false, 64);
        test_leanvec_top<leanvec::LeanDataset<float, leanvec::UsingLVQ<8>, Dyn, Dyn>>(
            false, 64
        );
    }
}
