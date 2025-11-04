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

#include "IndexSVSTrainingInfo.h"
#include "detail/TrainingInfoImpl.h"

namespace svs {
namespace runtime {

IndexSVSTrainingInfo::IndexSVSTrainingInfo(
    std::unique_ptr<svs::runtime::detail::TrainingInfoImpl> impl
) noexcept
    : impl_(std::move(impl)) {}

void IndexSVSTrainingInfo::destroy(IndexSVSTrainingInfo* impl) noexcept { delete impl; }

Status IndexSVSTrainingInfo::serialize(std::ostream& out) const noexcept {
    if (impl_ == nullptr) {
        return Status_Ok;
    }
    try {
        impl_->serialize(out);
    } catch (std::exception& e) {
        return Status{ErrorCode::IO_ERROR, e.what()};
    } catch (...) {
        return Status{ErrorCode::IO_ERROR, "Failed to serialize IndexSVSTrainingInfo"};
    }
    return Status_Ok;
}

Status IndexSVSTrainingInfo::deserialize(std::istream& in) noexcept {
    if (!impl_) {
        return Status_Ok;
    }
    try {
        impl_->deserialize(in);
    } catch (std::exception& e) {
        return Status{ErrorCode::IO_ERROR, e.what()};
    } catch (...) {
        return Status{ErrorCode::IO_ERROR, "Failed to deserialize IndexSVSTrainingInfo"};
    }
    return Status_Ok;
}

} // namespace runtime
} // namespace svs
