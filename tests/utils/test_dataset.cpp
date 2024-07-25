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

// svs-test
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// svs
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/core/graph.h"

// stl
#include <cstdint>
#include <filesystem>
#include <vector>

namespace test_dataset {

std::filesystem::path dataset_directory() {
    return svs_test::data_directory() / "test_dataset";
}

std::filesystem::path reference_directory() {
    return test_dataset::dataset_directory() / "reference";
}

std::filesystem::path reference_vecs_file() {
    return dataset_directory() / "known_f32.fvecs";
}

std::filesystem::path reference_svs_file() { return dataset_directory() / "known_f32.svs"; }

std::vector<std::vector<float>> reference_file_contents() {
    return std::vector<std::vector<float>>{{
        {-0.5297755, -0.46527258, -0.35637274, -0.08176492, 1.5503496, -0.7668221},
        {-2.4953504, 0.69067955, 1.4129586, 0.96996725, -1.0216018, 0.8098934},
        {-0.7779222, -1.1489166, 1.8277988, -0.3818305, -0.014146144, -1.0575522},
        {-0.07507572, 0.6534284, -1.1132482, 0.4399589, 0.20736118, -0.70264465},
        {1.0966406, -0.7609801, -1.2466722, 0.82666475, 0.12550473, 1.760032},
    }};
}

std::filesystem::path data_svs_file() { return dataset_directory() / "data_f32.svs"; }

std::filesystem::path graph_file() { return dataset_directory() / "graph_128.svs"; }

std::filesystem::path vamana_config_file() {
    return dataset_directory() / "vamana_config.toml";
}

std::filesystem::path metadata_file() { return dataset_directory() / "metadata.svs"; }

std::filesystem::path query_file() { return dataset_directory() / "queries_f32.fvecs"; }

std::filesystem::path groundtruth_euclidean_file() {
    return dataset_directory() / "groundtruth_euclidean.ivecs";
}

std::filesystem::path groundtruth_mip_file() {
    return dataset_directory() / "groundtruth_mip.ivecs";
}

std::filesystem::path groundtruth_cosine_file() {
    return dataset_directory() / "groundtruth_cosine.ivecs";
}

svs::data::SimpleData<float> queries() { return svs::load_data<float>(query_file()); }
svs::data::SimpleData<uint32_t> groundtruth_euclidean() {
    return svs::load_data<uint32_t>(groundtruth_euclidean_file());
}
svs::data::SimpleData<uint32_t> groundtruth_mip() {
    return svs::load_data<uint32_t>(groundtruth_mip_file());
}
svs::data::SimpleData<uint32_t> groundtruth_cosine() {
    return svs::load_data<uint32_t>(groundtruth_cosine_file());
}
svs::data::SimpleData<float> data_f32() {
    return svs::load_data<float, svs::Dynamic>(data_svs_file());
}

svs::data::BlockedData<float> data_blocked_f32() {
    return svs::data::BlockedData<float>::load(data_svs_file());
}
svs::graphs::SimpleGraph<uint32_t> graph() {
    return svs::graphs::SimpleGraph<uint32_t>::load(graph_file());
}

svs::graphs::SimpleBlockedGraph<uint32_t> graph_blocked() {
    return svs::graphs::SimpleBlockedGraph<uint32_t>::load(graph_file());
}

std::vector<uint32_t> expected_out_neighbors() {
    return std::vector<uint32_t>{
        64, 103, 118, 45, 34, 31, 64, 121, 128, 128, 128, 128, 46, 71, 115, 112};
}

// Helper to load the ground-truth for a given file.
svs::data::SimpleData<uint32_t> load_groundtruth(svs::DistanceType distance) {
    switch (distance) {
        case svs::DistanceType::L2: {
            return test_dataset::groundtruth_euclidean();
        }
        case svs::DistanceType::MIP: {
            return test_dataset::groundtruth_mip();
        }
        case svs::DistanceType::Cosine: {
            return test_dataset::groundtruth_cosine();
        }
    }
    throw ANNEXCEPTION("Unhandled distance!");
}

} // namespace test_dataset
