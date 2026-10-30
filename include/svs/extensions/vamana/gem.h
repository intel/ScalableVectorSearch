/*
 * Copyright 2026 Intel Corporation
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

// index/vamana depends on core/distance, not the reverse, so these overrides live here
// rather than in core/distance/gem.h.
#include "svs/core/distance/gem.h"
#include "svs/index/vamana/extensions.h"
#include "svs/index/vamana/prune.h"

namespace svs::index::vamana {

// qEMD is "smaller is better" like L2 (std::less<>), so share its MRNG pruning
// bookkeeping rather than IP/cosine's.
template <> struct PruneStrategy<svs::distance::QEmd> {
    using type = ProgressivePruneStrategy;
};

// qCH is also std::less<>; the MVG ablation (--metric qch) builds the graph with QChamfer
// directly, so it needs the same pruning strategy as QEmd.
template <> struct PruneStrategy<svs::distance::QChamfer> {
    using type = ProgressivePruneStrategy;
};

namespace extensions {

// DefaultBuildAdaptor default-constructs its distance_ member, which drops the
// CentroidTable svs::distance::QEmd holds; forward the real functor through aggregate-init
// instead.
inline DefaultBuildAdaptor<svs::distance::QEmd> svs_invoke(
    svs::tag_t<build_adaptor>, const auto& /*data*/, const svs::distance::QEmd& distance
) {
    return DefaultBuildAdaptor<svs::distance::QEmd>{distance};
}

// Same reason as the QEmd overload above: QChamfer also holds a CentroidTable that a
// default-constructed distance_ would drop.
inline DefaultBuildAdaptor<svs::distance::QChamfer> svs_invoke(
    svs::tag_t<build_adaptor>, const auto& /*data*/, const svs::distance::QChamfer& distance
) {
    return DefaultBuildAdaptor<svs::distance::QChamfer>{distance};
}

} // namespace extensions
} // namespace svs::index::vamana
