/*
 * Copyright 2023 Intel Corporation
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
