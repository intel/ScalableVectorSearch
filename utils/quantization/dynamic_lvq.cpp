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

// svsmain
#include "svsmain.h"

// svs
#include "svs/extensions/vamana/lvq.h"
#include "svs/index/vamana/dynamic_index.h"
#include "svs/misc/dynamic_helper.h"

namespace lvq = svs::quantization::lvq;

template <std::integral I> I div(I i, float fraction) {
    return svs::lib::narrow<I>(std::floor(svs::lib::narrow<float>(i) * fraction));
}

int svs_main(std::vector<std::string> args) {
    size_t i = 1;
    const auto& path = args.at(i++);
    const auto& query_path = args.at(i++);

    const float modify_fraction = 0.125;
    size_t num_threads = 10;

    // Load the data we're going to compress.
    auto queries = svs::VectorDataLoader<float>(query_path).load();
    auto base_data = svs::VectorDataLoader<float>(path).load();
    auto num_points = base_data.size();

    auto reference = svs::misc::
        ReferenceDataset<uint32_t, float, svs::Dynamic, svs::distance::DistanceL2>{
            std::move(base_data),
            svs::distance::DistanceL2(),
            num_threads,
            div(num_points, 0.015625 * modify_fraction),
            10,
            queries};

    // Allocate the dataset.
    // auto compressor = lvq::OneLevelWithBias<8>(lvq::Reload(""));
    auto compressor = lvq::TwoLevelWithBias<8, 8>(lvq::Reload(""));
    auto [data, ids] = reference.generate(10'000);
    auto lvq_dataset = compressor.compress(data, svs::data::BlockedBuilder());

    size_t max_degree = 32;
    auto parameters = svs::index::vamana::VamanaBuildParameters{
        1.2, max_degree, 2 * max_degree, 1000, max_degree - 4, true};

    auto index = svs::index::vamana::MutableVamanaIndex{
        parameters, std::move(lvq_dataset), ids, svs::distance::DistanceL2(), num_threads};

    reference.add_points(index, 10'000);
    reference.delete_points(index, 10'000);
    index.consolidate();
    index.compact();
    return 0;
}

SVS_DEFINE_MAIN();
