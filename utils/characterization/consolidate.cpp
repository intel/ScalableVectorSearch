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

#include "svs/index/vamana/consolidate.h"
#include "svs/index/vamana/index.h"
#include "svs/lib/timing.h"

#include "svsmain.h"

// tsl
#include "tsl/robin_set.h"

// stdlib
#include <random>
#include <unordered_set>

using Idx = uint32_t;
using Eltype = svs::Float16;
// const size_t Extent = svs::Dynamic;
const size_t Extent = 96;

int svs_main(std::vector<std::string> args) {
    // Unpack arguments.
    const auto& graph_path = args.at(1);
    const auto& data_path = args.at(2);
    auto delete_percent = std::stof(args.at(3));
    auto nthreads = std::stoull(args.at(4));

    if (delete_percent < 0 || delete_percent > 1) {
        throw ANNEXCEPTION(
            "Delete percent must be between 0 and 1. Instead, got ", delete_percent, "!"
        );
    }

    // Load the graph.
    std::cout << "Loading Graph" << std::endl;
    auto graph =
        svs::graphs::SimpleGraph<Idx>::load(graph_path, svs::HugepageAllocator<Idx>());

    std::cout << "Loading Data" << std::endl;
    auto data = svs::VectorDataLoader<Eltype, Extent>(data_path).load();

    // Create a random number generator.
    std::cout << "Generating Indices" << std::endl;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<Idx> distribution{0, svs::lib::narrow<Idx>(data.size())};

    tsl::robin_set<Idx> indices_to_delete{};
    auto target_count = svs::lib::narrow<size_t>(
        std::lround(delete_percent * svs::lib::narrow<float>(data.size()))
    );

    while (indices_to_delete.size() < target_count) {
        indices_to_delete.insert(distribution(rng));
    }

    auto threadpool = svs::threads::NativeThreadPool(nthreads);

    // Now, perform the deletion.
    std::cout << "Consolidating Graph" << std::endl;
    svs::distance::DistanceL2 distance{};
    auto tic = svs::lib::now();
    svs::index::vamana::consolidate(
        graph,
        data,
        threadpool,
        graph.max_degree(),
        750,
        1.2f,
        distance,
        [&](const auto& i) { return indices_to_delete.contains(i); }
    );
    auto runtime = svs::lib::time_difference(tic);
    std::cout << "Consolidation took " << runtime << " seconds." << std::endl;

    return 0;
}

SVS_DEFINE_MAIN();
