/*
 * Copyright 2024 Intel Corporation
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
