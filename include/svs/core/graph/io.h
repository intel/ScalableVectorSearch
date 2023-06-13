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

#include "svs/concepts/graph.h"
#include "svs/core/allocator.h"
#include "svs/core/data/io.h"
#include "svs/core/graph/graph.h"
#include "svs/core/io.h"
#include "svs/lib/uuid.h"

namespace svs {
namespace io {

template <typename Ret, typename File, typename Builder>
Ret load_graph(const File& file, const Builder& builder) {
    using Idx = typename Ret::index_type;
    return Ret{load_dataset<Idx, Dynamic>(file, builder)};
}

///
/// Simple
///

// Reader rvalue reference overload.
template <typename Idx, typename File, typename Allocator>
graphs::SimpleGraph<Idx> load_simple_graph(const File& file, const Allocator& allocator) {
    return load_graph<graphs::SimpleGraph<Idx>>(file, data::PolymorphicBuilder{allocator});
}

///
/// Blocked
///

template <typename Idx, typename File>
graphs::SimpleBlockedGraph<Idx> load_blocked_graph(const File& file) {
    return load_graph<graphs::SimpleBlockedGraph<Idx>>(file, data::BlockedBuilder());
}

/////
///// Graph Saving
/////

template <typename Idx>
void save(
    const graphs::SimpleGraph<Idx>& graph,
    const NativeFile& file,
    const lib::UUID& uuid = lib::UUID(lib::ZeroInitializer())
) {
    save(graph.get_data(), file, uuid);
}
} // namespace io
} // namespace svs
