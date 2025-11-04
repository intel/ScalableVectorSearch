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

#pragma once
#include "IndexSVSImplDefs.h"

#include <iostream>
#include <memory>

namespace svs {
namespace runtime {

namespace detail {
struct TrainingInfoImpl;
}

struct SVS_RUNTIME_API IndexSVSTrainingInfo {
    IndexSVSTrainingInfo() noexcept = default;

    IndexSVSTrainingInfo(std::unique_ptr<svs::runtime::detail::TrainingInfoImpl> impl
    ) noexcept;

    static void destroy(IndexSVSTrainingInfo* impl) noexcept;
    virtual ~IndexSVSTrainingInfo();

    Status serialize(std::ostream& out) const noexcept;
    Status deserialize(std::istream& in) noexcept;

  protected:
    std::unique_ptr<svs::runtime::detail::TrainingInfoImpl> impl_{nullptr};
};

} // namespace runtime
} // namespace svs
