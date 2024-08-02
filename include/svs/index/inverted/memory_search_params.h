/**
 *    Copyright (C) 2023, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

#pragma once

// svs
#include "svs/index/vamana/search_params.h"
#include "svs/lib/saveload.h"
#include "svs/lib/version.h"

namespace svs::index::inverted {

struct InvertedSearchParameters {
  public:
    vamana::VamanaSearchParameters primary_parameters_{};
    double refinement_epsilon_ = 1.0;

  public:
    InvertedSearchParameters() = default;

    InvertedSearchParameters(
        const vamana::VamanaSearchParameters primary_parameters, double refinement_epsilon
    )
        : primary_parameters_{primary_parameters}
        , refinement_epsilon_{refinement_epsilon} {}

    SVS_CHAIN_SETTER_(InvertedSearchParameters, primary_parameters);
    SVS_CHAIN_SETTER_(InvertedSearchParameters, refinement_epsilon);

    ///// Saving and Loading.
    static constexpr lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "inverted_search_parameters";
    lib::SaveTable save() const {
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(primary_parameters), SVS_LIST_SAVE_(refinement_epsilon)}
        );
    }

    static InvertedSearchParameters load(const lib::ContextFreeLoadTable& table) {
        return InvertedSearchParameters{
            SVS_LOAD_MEMBER_AT_(table, primary_parameters),
            SVS_LOAD_MEMBER_AT_(table, refinement_epsilon)};
    }

    constexpr friend bool
    operator==(const InvertedSearchParameters&, const InvertedSearchParameters&) = default;
};

} // namespace svs::index::inverted
