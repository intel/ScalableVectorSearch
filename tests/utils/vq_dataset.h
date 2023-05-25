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

// svs
#include "svs/core/data.h"

// stl
#include <filesystem>
#include <vector>

namespace vq_dataset {

template <typename T> using SPD = svs::data::SimplePolymorphicData<T, svs::Dynamic>;

struct VQDataset {
    // Datasets
    SPD<float> data_f32;
    SPD<svs::Float16> data_f16;
    SPD<float> queries;
    // Pre-computed quantities
    std::vector<float> means;
    std::vector<float> variances;
    std::vector<float> minimums;
    std::vector<float> maximums;
};

///
/// Return a path to the vector quantization test datasets files.
///
std::filesystem::path directory();

///
/// Load the vector quantiation test datafiles located in the directory `dir`.
///
VQDataset load(const std::filesystem::path& dir = directory());

} // namespace vq_dataset
