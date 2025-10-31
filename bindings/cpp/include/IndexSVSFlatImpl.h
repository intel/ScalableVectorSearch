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

#include "IndexSVSImplDefs.h"
#include <istream>
#include <memory>
#include <ostream>

namespace svs {
class Flat;

namespace runtime {
class SVS_RUNTIME_API IndexSVSFlatImpl {
  public:
    static IndexSVSFlatImpl* build(size_t dim, MetricType metric) noexcept;
    static void destroy(IndexSVSFlatImpl* impl) noexcept;

    Status add(size_t n, const float* x) noexcept;
    void reset() noexcept;

    Status search(size_t n, const float* x, size_t k, float* distances, size_t* labels)
        const noexcept;

    Status serialize(std::ostream& out) const noexcept;

    Status deserialize(std::istream& in) noexcept;

  private:
    IndexSVSFlatImpl(size_t dim, MetricType metric);
    ~IndexSVSFlatImpl();
    Status init_impl(size_t n, const float* x) noexcept;

    MetricType metric_type_;
    size_t dim_;
    std::unique_ptr<svs::Flat> impl{nullptr};
};
} // namespace runtime
} // namespace svs
