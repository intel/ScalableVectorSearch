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
#include "svs/core/distance.h"
#include "svs/core/graph.h"

#include "tests/utils/utils.h"

namespace test_dataset {

///// Paths
// The directory containing the reference dataset.
std::filesystem::path dataset_directory();
// The directory containint test reference results.
std::filesystem::path reference_directory();
// A fvecs file with known contents.
std::filesystem::path reference_vecs_file();
// The same file as the "vecs" file but encoded in the SVS format.
std::filesystem::path reference_svs_file();
// The expected contents of the reference file.
std::vector<std::vector<float>> reference_file_contents();

// The test data encoded in the "svs" format.
std::filesystem::path data_svs_file();
// Test graph in the "svs" format.
std::filesystem::path graph_file();
// Index metadata file.
std::filesystem::path vamana_config_file();

std::filesystem::path metadata_file();

// Test query data in the "fvecs" format.
std::filesystem::path query_file();
// Groundtruth of for the queries with respect to the dataset using the euclidean distance.
std::filesystem::path groundtruth_euclidean_file();
// Groundtruth of for the queries with respect to the dataset using the MIP distance.
std::filesystem::path groundtruth_mip_file();
// Groundtruth of for the queries with respect to the dataset using cosine similarity.
std::filesystem::path groundtruth_cosine_file();

// LeanVec data and query matrices in "fvecs" format.
std::filesystem::path leanvec_data_matrix_file();
std::filesystem::path leanvec_query_matrix_file();

///// Helper Functions
svs::data::SimpleData<float> queries();
svs::data::SimpleData<uint32_t> groundtruth_euclidean();
svs::data::SimpleData<uint32_t> groundtruth_mip();
svs::data::SimpleData<uint32_t> groundtruth_cosine();
svs::data::SimpleData<float> data_f32();

svs::data::BlockedData<float> data_blocked_f32();

svs::graphs::SimpleGraph<uint32_t> graph();
svs::graphs::SimpleBlockedGraph<uint32_t> graph_blocked();

/// Helper to load the ground-truth for a given file.
svs::data::SimpleData<uint32_t> load_groundtruth(svs::DistanceType distance);

/// Helpers to load LeanVec OOD matrices
template <size_t D = svs::Dynamic> svs::data::SimpleData<float, D> leanvec_data_matrix() {
    return svs::load_data<float, D>(leanvec_data_matrix_file());
}
template <size_t D = svs::Dynamic> svs::data::SimpleData<float, D> leanvec_query_matrix() {
    return svs::load_data<float, D>(leanvec_query_matrix_file());
}

///
/// @brief Return a reference to the last `queries_in_test_set` entries in `queries`.
///
/// **NOTE**: The returned item is a view into the orignal queries. As such, is `queries`'s
/// destructor runs, then the returned view will be left danglinc.
///
template <typename T, size_t N, typename Allocator>
svs::data::ConstSimpleDataView<T, N> get_test_set(
    const svs::data::SimpleData<T, N, Allocator>& queries, size_t queries_in_test_set
) {
    const size_t n_queries = queries.size();
    if (queries_in_test_set > n_queries) {
        throw ANNEXCEPTION(
            "Requested number of queries in test set ({}) exceeds the actual number of "
            "queries ({})!",
            queries_in_test_set,
            n_queries
        );
    }
    if (queries.dimensions() == 0) {
        throw ANNEXCEPTION("Cannot extract test set from queries with 0 dimensions!");
    }

    const auto* ptr = &queries.get_datum(n_queries - queries_in_test_set).front();
    return svs::data::ConstSimpleDataView<T, N>(
        ptr, queries_in_test_set, queries.dimensions()
    );
}

///// Graph Stats
// The expected out degrees of the first few entries in the graph.
std::vector<uint32_t> expected_out_neighbors();
const size_t GRAPH_MAX_DEGREE = 128;

// Data Stats
const size_t NUM_DIMENSIONS = 128;
const size_t VECTORS_IN_DATA_SET = 10000;
const float SUM_OF_FIRST_VECTOR = -523.0f;
const float SUM_OF_SECOND_VECTOR = -79.0f;
const float SUM_OF_FIRST_TWO_VECTORS = SUM_OF_FIRST_VECTOR + SUM_OF_SECOND_VECTOR;
const double SUM_OF_ALL_VECTORS = 28887.0f;

} // namespace test_dataset
