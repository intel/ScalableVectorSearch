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
    lib::SaveTable save() const {
        return lib::SaveTable(
            save_version,
            {SVS_LIST_SAVE_(clustering_parameters), SVS_LIST_SAVE_(primary_parameters)}
        );
    }

    static InvertedBuildParameters
    load(const toml::table& table, const lib::Version& version) {
        if (version != save_version) {
            throw ANNEXCEPTION("Version mismatch!");
        }

        return InvertedBuildParameters(
            SVS_LOAD_MEMBER_AT_(table, clustering_parameters),
            SVS_LOAD_MEMBER_AT_(table, primary_parameters)
        );
    }
};

} // namespace svs::index::inverted
