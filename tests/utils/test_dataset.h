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

#include <cstdint>
#include <filesystem>
#include <vector>

#include "svs/core/data.h"
#include "svs/core/graph.h"

#include "tests/utils/utils.h"

namespace test_dataset {

///// Summary
//
// -- Paths --
// reference_vecs_file() -> std::filesystem::path
// reference_svs_file() -> std::filesystem::path
// dataset_directory() -> std::filesystem::path
// data_svs_file() -> std::filesystem::path
// graph_file() -> std::filesystem::path
// metadata_file() -> std::filesystem::path
// query_file() -> std::filesystem::path
// groundtruth_euclidean_file() -> std::filesystem::path
// groundtruth_mip_file() -> std::filesystem::path
// groundtruth_cosine_file() -> std::filesystem::path
//
// -- Reference Contents --
// reference_file_contents() -> std::vector<std::vector<float>>
//
// -- Loading --
// queries() -> svs::QueryData<float>
// groundtruth_euclidean() -> svs::QueryData<uint32_t>
// groundtruth_mip() -> svs::QueryData<uint32_t>
// groundtruth_cosine() -> svs::QueryData<uint32_t>
// data_f32()
// graph()
//
// -- Graph Stats --
// expected_out_neighbors() -> std::vector<uint32_t>
// const size_t GRAPH_MAX_DEGREE
//
// -- Data Stats --
// const size_t NUM_DIMENSIONS
// const size_t VECTORS_IN_DATA_SET
// const float SUM_OF_FIRST_VECTOR
// const float SUM_OF_SECOND_VECTOR
// const float SUM_OF_FIRST_TWO_VECTORS
// const double SUM_OF_ALL_VECTORS

///// Paths
inline std::filesystem::path dataset_directory() {
    return svs_test::data_directory() / "test_dataset";
}

// A fvecs file with known contents.
inline std::filesystem::path reference_vecs_file() {
    return dataset_directory() / "known_f32.fvecs";
}

inline std::filesystem::path reference_svs_file() {
    return dataset_directory() / "known_f32.svs";
}

inline std::vector<std::vector<float>> reference_file_contents() {
    return std::vector<std::vector<float>>{{
        {-0.5297755, -0.46527258, -0.35637274, -0.08176492, 1.5503496, -0.7668221},
        {-2.4953504, 0.69067955, 1.4129586, 0.96996725, -1.0216018, 0.8098934},
        {-0.7779222, -1.1489166, 1.8277988, -0.3818305, -0.014146144, -1.0575522},
        {-0.07507572, 0.6534284, -1.1132482, 0.4399589, 0.20736118, -0.70264465},
        {1.0966406, -0.7609801, -1.2466722, 0.82666475, 0.12550473, 1.760032},
    }};
}

// The test data encoded in the "svs" format.
inline std::filesystem::path data_svs_file() {
    return dataset_directory() / "data_f32.svs";
}

// Test graph in the "svs" format.
inline std::filesystem::path graph_file() { return dataset_directory() / "graph_128.svs"; }

// Index metadata file.
inline std::filesystem::path vamana_config_file() {
    return dataset_directory() / "vamana_config.toml";
}

inline std::filesystem::path metadata_file() {
    return dataset_directory() / "metadata.svs";
}

// Test query data in the "fvecs" format.
inline std::filesystem::path query_file() {
    return dataset_directory() / "queries_f32.fvecs";
}

// Groundtruth of for the queries with respect to the dataset using the euclidean distance.
inline std::filesystem::path groundtruth_euclidean_file() {
    return dataset_directory() / "groundtruth_euclidean.ivecs";
}

// Groundtruth of for the queries with respect to the dataset using the MIP distance.
inline std::filesystem::path groundtruth_mip_file() {
    return dataset_directory() / "groundtruth_mip.ivecs";
}

// Groundtruth of for the queries with respect to the dataset using cosine similarity.
inline std::filesystem::path groundtruth_cosine_file() {
    return dataset_directory() / "groundtruth_cosine.ivecs";
}

///// Helper Functions
inline auto queries() { return svs::load_data<float>(query_file()); }
inline auto groundtruth_euclidean() {
    return svs::load_data<uint32_t>(groundtruth_euclidean_file());
}
inline auto groundtruth_mip() { return svs::load_data<uint32_t>(groundtruth_mip_file()); }
inline auto groundtruth_cosine() {
    return svs::load_data<uint32_t>(groundtruth_cosine_file());
}
inline auto data_f32() { return svs::load_data<float, svs::Dynamic>(data_svs_file()); }
inline auto data_blocked_f32() {
    return svs::data::BlockedData<float>::load(data_svs_file());
}
inline auto graph() { return svs::graphs::SimpleGraph<uint32_t>::load(graph_file()); }
inline auto graph_blocked() {
    return svs::graphs::SimpleBlockedGraph<uint32_t>::load(graph_file());
}

///// Graph Stats
// The expected out degrees of the first few entries in the graph.
inline std::vector<uint32_t> expected_out_neighbors() {
    return std::vector<uint32_t>{
        64, 103, 118, 45, 34, 31, 64, 121, 128, 128, 128, 128, 46, 71, 115, 112};
}
const size_t GRAPH_MAX_DEGREE = 128;

// Data Stats
const size_t NUM_DIMENSIONS = 128;
const size_t VECTORS_IN_DATA_SET = 10000;
const float SUM_OF_FIRST_VECTOR = -523.0f;
const float SUM_OF_SECOND_VECTOR = -79.0f;
const float SUM_OF_FIRST_TWO_VECTORS = SUM_OF_FIRST_VECTOR + SUM_OF_SECOND_VECTOR;
const double SUM_OF_ALL_VECTORS = 28887.0f;

} // namespace test_dataset
