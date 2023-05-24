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
#include "svs/concepts/data.h"
#include "svs/core/io.h"

#include "svs/lib/array.h"
#include "svs/lib/exception.h"
#include "svs/lib/memory.h"
#include "svs/lib/meta.h"
#include "svs/lib/misc.h"

namespace svs::io {

// Dispatch tags to control the loading and saving pipeline.
constexpr lib::PriorityTag<2> default_populate_tag = lib::PriorityTag<2>();
constexpr lib::PriorityTag<2> default_save_tag = lib::PriorityTag<2>();

// Generic dataset population.
// TODO (long term): Constrain this to vector-based datasets?
template <data::MemoryDataset Data, typename File>
void populate_impl(Data& data, const File& file, lib::PriorityTag<0> /*tag*/) {
    using T = typename Data::element_type;
    auto reader = file.reader(lib::meta::Type<T>());
    size_t i = 0;
    for (auto v : reader) {
        data.set_datum(i, v);
        ++i;
    }
}

// Intercept the native file to perform dispatch on the actual file type.
template <data::MemoryDataset Data>
void populate_impl(Data& data, const NativeFile& file, lib::PriorityTag<1> tag) {
    file.resolve([&](const auto& resolved_file) {
        populate_impl(data, resolved_file, tag.next());
    });
}

///
/// @brief Populate the entries of `data` with the contents of `file`.
///
template <data::MemoryDataset Data, typename File>
void populate(Data& data, const File& file) {
    populate_impl(data, file, default_populate_tag);
}

/////
///// Saving
/////

template <data::ImmutableMemoryDataset Dataset, typename File>
void save_impl(
    const Dataset& data,
    const File& file,
    const lib::UUID& uuid = lib::ZeroUUID,
    lib::PriorityTag<0> SVS_UNUSED(tag) = lib::PriorityTag<0>()
) {
    auto writer = file.writer(data.dimensions(), uuid);
    for (size_t i = 0; i < data.size(); ++i) {
        writer << data.get_datum(i);
    }
}

template <data::ImmutableMemoryDataset Dataset, typename File>
void save(const Dataset& data, const File& file, const lib::UUID& uuid = lib::ZeroUUID) {
    save_impl(data, file, uuid, default_save_tag);
}

///
/// @brief Save the dataset as a "*vecs" file.
///
/// @param data The dataset to save.
/// @param path The file path to where the data will be saved.
///
template <data::ImmutableMemoryDataset Dataset>
void save_vecs(const Dataset& data, const std::filesystem::path& path) {
    auto file = vecs::VecsFile{path};
    auto writer = file.writer(data.dimensionsions());
    for (size_t i = 0; i < data.size(); ++i) {
        writer << data.get_datum(i);
    }
}
} // namespace svs::io
