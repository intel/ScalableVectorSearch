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

#pragma once

#include "svs/index/inverted/clustering.h"
#include "svs/index/vamana/build_params.h"
#include "svs/lib/saveload.h"

namespace svs::index::inverted {

struct InvertedBuildParameters {
  public:
    /// Parameters of the clustering process.
    inverted::ClusteringParameters clustering_parameters_;
    /// Construction parameters for the primary index.
    vamana::VamanaBuildParameters primary_parameters_;

  public:
    InvertedBuildParameters() = default;
    InvertedBuildParameters(
        const inverted::ClusteringParameters& clustering_parameters,
        const vamana::VamanaBuildParameters& primary_parameters
    )
        : clustering_parameters_{clustering_parameters}
        , primary_parameters_{primary_parameters} {}

    // Comparison
    friend constexpr bool
    operator==(const InvertedBuildParameters&, const InvertedBuildParameters&) = default;

    // Saving
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "inverted_build_parameters";
    lib::SaveTable save() const {
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(clustering_parameters), SVS_LIST_SAVE_(primary_parameters)}
        );
    }

    static InvertedBuildParameters load(const lib::ContextFreeLoadTable& table) {
        return InvertedBuildParameters(
            SVS_LOAD_MEMBER_AT_(table, clustering_parameters),
            SVS_LOAD_MEMBER_AT_(table, primary_parameters)
        );
    }
};

} // namespace svs::index::inverted
