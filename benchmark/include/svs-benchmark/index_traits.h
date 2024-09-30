/*
 * Copyright (C) 2024 Intel Corporation
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

namespace svsbenchmark {

// Customize this data structure with various index imp;lementations
template <typename Index> struct IndexTraits; //{
///// Basic Requiresments
// using index_type = Index;
// using config_type = ...;
// using state_type = ...;
// static std::string name();
//
///// Search Requirements
// template<svs::data::ImmutableMemoryDataset Queries>
// static auto search(index_type&, const Queries&, size_t num_neighbors, const
// config_type&);
//
// template<svs::data::ImmutableMemoryDataset Queries, typename GroundTruth>
// static config_type calibrate(
//      index_type&,
//      const Queries&,
//      const GroundTruth&,
//      size_t num_neighbors,
//      size_t target_recall
// );
//
// static state_type report_state(const index_type&);
//
// static void apply_config(index_type&, const config_type&);
//
///// Dynamic Build Requirements
//
// template<svs::data::ImmutableMemoryDataset Points>
// static void add_points(index_type&, const Points& points, const std::vector<size_t>&
// ids);
//
// static void delete_points(index_type&, const std::vector<size_t>& ids);
//
// static void consolidate(index_type&);
// };

template <typename Index> using config_type = typename IndexTraits<Index>::config_type;
template <typename Index> using state_type = typename IndexTraits<Index>::state_type;

} // namespace svsbenchmark
