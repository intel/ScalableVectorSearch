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
