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
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "svs/core/data.h"
#include "svs/core/recall.h"
#include "svsmain.h"

const std::string HELP =
    R"(
   compute_recall groundtruth results

Compute the "k-recall at k" where "k" is the number of neighbors for each entry
in results.

The groundtruth and results must have the same number of vectors.
)";

int svs_main(std::vector<std::string> args) {
    if (args.size() != 3) {
        std::cout << HELP << std::endl;
        return 1;
    }

    const auto& groundtruth_path = args[1];
    const auto& results_path = args[2];
    auto recall = svs::k_recall_at_n(
        svs::load_data<uint32_t>(groundtruth_path), svs::load_data<uint32_t>(results_path)
    );
    std::cout << recall << '\n';
    return 0;
}

SVS_DEFINE_MAIN();
