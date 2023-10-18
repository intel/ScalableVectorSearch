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

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "svs/core/data.h"
#include "svs/extensions/vamana/lvq.h"
#include "svs/orchestrators/vamana.h"

#include "svsmain.h"

namespace lvq = svs::quantization::lvq;

const size_t N = 96;
using Distance = svs::distance::DistanceL2;

int svs_main(std::vector<std::string> args) {
    const auto& data_path(args[1]);

    const size_t num_threads = 10;
    const size_t max_degree = 64;

    auto parameters = svs::index::vamana::VamanaBuildParameters{
        1.2, max_degree, 100, 1000, max_degree - 4, true};

    auto index = svs::index::vamana::auto_build(
        parameters,
        lvq::OneLevelWithBias<8, N>(svs::VectorDataLoader<svs::Float16, N>(data_path), 0),
        // lvq::TwoLevelWithBias<4, 4, N>(svs::VectorDataLoader<svs::Float16, N>(data_path),
        // 0),
        Distance(),
        num_threads,
        svs::DRAM()
    );

    return 0;
}

SVS_DEFINE_MAIN();
