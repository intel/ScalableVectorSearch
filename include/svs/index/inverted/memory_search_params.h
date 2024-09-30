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
