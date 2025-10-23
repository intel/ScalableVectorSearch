/*
 * Copyright 2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
