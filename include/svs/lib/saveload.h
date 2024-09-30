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
// Utilities for saving and loading objects.
#pragma once

// svs
#include "svs/lib/saveload/bootstrap.h"
#include "svs/lib/saveload/core.h"
#include "svs/lib/saveload/load.h"
#include "svs/lib/saveload/save.h"

namespace svs::lib {

///// testing
template <typename T>
bool test_self_save_load(const T& x, const std::filesystem::path& dir) {
    svs::lib::save_to_disk(x, dir);
    auto y = svs::lib::load_from_disk<T>(dir);
    return x == y;
}

template <typename T> bool test_self_save_load_context_free(const T& x) {
    const auto& serialized = svs::lib::save(x);
    auto y = svs::lib::load<T>(svs::lib::node_view(serialized));
    return x == y;
}

/////
///// Utility Macros
/////

// Expected Transformation:
// SVS_LIST_SAVE_(x, args...) -> {"x", svs::lib::save(x_, args...)}
#define SVS_LIST_SAVE_(name, ...)                     \
    {                                                 \
#name, svs::lib::save(name##_, ##__VA_ARGS__) \
    }

// Expected Transformation:
// SVS_INSERT_SAVE_(table, x, args...)
//  -> table.insert("x", svs::lib::save(x_, args...))
#define SVS_INSERT_SAVE_(table, name, ...) \
    table.insert(#name, svs::lib::save(name##_, ##__VA_ARGS__))

// Expected Transformation:
// SVS_LOAD_MEMBER_AT_(table, x, args...)
//  -> svs::lib::load_at<std::decay_t<decltype(x_)>>(table, x_, args...)
#define SVS_LOAD_MEMBER_AT_(table, name, ...) \
    svs::lib::load_at<std::decay_t<decltype(name##_)>>(table, #name, ##__VA_ARGS__)

// Non-underscored version

// Expected Transformation:
// SVS_LIST_SAVE_(x, args...) -> {"x", svs::lib::save(x, args...)}
#define SVS_LIST_SAVE(name, ...)                   \
    {                                              \
#name, svs::lib::save(name, ##__VA_ARGS__) \
    }

// Expected Transformation:
// SVS_INSERT_SAVE_(table, x, args...)
//  -> table.insert("x", svs::lib::save(x, args...))
#define SVS_INSERT_SAVE(table, name, ...) \
    table.insert(#name, svs::lib::save(name, ##__VA_ARGS__))

// Expected Transformation:
// SVS_LOAD_MEMBER_AT(table, x, args...)
//  -> svs::lib::load_at<std::decay_t<decltype(x_)>>(table, x_, args...)
#define SVS_LOAD_MEMBER_AT(table, name, ...) \
    svs::lib::load_at<std::decay_t<decltype(name)>>(table, #name, ##__VA_ARGS__)

} // namespace svs::lib
