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

// header
#include "tests/utils/vq_dataset.h"

// svs
#include "svs/core/data.h"
#include "svs/lib/readwrite.h"

// tests
#include "tests/utils/utils.h"

// stl
#include <filesystem>
#include <fstream>
#include <vector>

namespace vq_dataset {

std::filesystem::path directory() { return svs_test::data_directory() / "vq_dataset"; }

namespace {
template <typename T>
void read_binary(const std::filesystem::path& path, std::vector<T>& v) {
    auto stream = std::ifstream(path, std::ios_base::in | std::ios_base::binary);
    svs::lib::read_binary(stream, v);
}
} // namespace

VQDataset load(const std::filesystem::path& dir) {
    auto data_f32 = svs::io::load_dataset<float, svs::Dynamic>(
        svs::io::v1::NativeReader<float>(dir / "data_f32.svs")
    );
    auto data_f16 = svs::io::load_dataset<svs::Float16, svs::Dynamic>(
        svs::io::v1::NativeReader<svs::Float16>(dir / "data_f16.svs")
    );
    auto queries = svs::io::load_dataset<float, svs::Dynamic>(
        svs::io::v1::NativeReader<float>(dir / "queries.svs")
    );

    size_t ndims = data_f32.dimensions();
    std::vector<float> means(ndims);
    std::vector<float> variances(ndims);
    std::vector<float> minimums(ndims);
    std::vector<float> maximums(ndims);

    read_binary(dir / "means.bin", means);
    read_binary(dir / "variances.bin", variances);
    read_binary(dir / "minimums.bin", minimums);
    read_binary(dir / "maximums.bin", maximums);

    return VQDataset{
        std::move(data_f32),
        std::move(data_f16),
        std::move(queries),
        std::move(means),
        std::move(variances),
        std::move(minimums),
        std::move(maximums)};
}

} // namespace vq_dataset
