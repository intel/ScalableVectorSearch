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
