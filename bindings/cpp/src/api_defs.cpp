/*
 * Copyright 2025 Intel Corporation
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

#include <svs/runtime/api_defs.h>

#include <cassert>
#include <cstring>

namespace svs {
namespace runtime {
void Status::store_message(const char* msg) noexcept {
    assert(msg != nullptr);
    try {
        auto len = std::strlen(msg);
        message_storage_ = new char[len + 1];
        std::strcpy(message_storage_, msg);
    } catch (...) {
        // In case of any error, leave message_storage_ as nullptr
        if (message_storage_) {
            delete[] message_storage_;
            message_storage_ = nullptr;
        }
        return;
    }
}

void Status::destroy_message() noexcept {
    assert(message_storage_ != nullptr);
    delete[] message_storage_;
    message_storage_ = nullptr;
}
} // namespace runtime
} // namespace svs