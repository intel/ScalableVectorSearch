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
    auto queries = svs::data::SimpleData<float>::load(query_path);
    auto base_data = svs::data::SimpleData<float>::load(path);
    auto num_points = base_data.size();

    auto reference = svs::misc::
        ReferenceDataset<uint32_t, float, svs::Dynamic, svs::distance::DistanceL2>{
            std::move(base_data),
            svs::distance::DistanceL2(),
            num_threads,
            div(num_points, 0.015625 * modify_fraction),
            10,
            queries,
            0x98af};

    // Allocate the dataset.
    auto [data, ids] = reference.generate(10'000);
    auto lvq_dataset = lvq::LVQDataset<
        8,
        8,
        svs::Dynamic,
        lvq::Sequential,
        svs::data::Blocked<svs::lib::Allocator<std::byte>>>::compress(data);

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
