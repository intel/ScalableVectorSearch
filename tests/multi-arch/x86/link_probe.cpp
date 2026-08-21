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

// A consumer that names every kernel the dispatch surface declares and nothing
// else, so a declared-but-uninstantiated kernel is an undefined symbol here.

#include "svs/core/distance/cosine.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/inner_product.h"
#include "svs/lib/static.h"
#include "svs/multi-arch/x86/preprocessor.h"

#include "host_levels.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>

namespace {

using svs::distance::AVX_AVAILABILITY;
using svs_test::host_satisfies;

// The longest fixed extent in the surface.
constexpr size_t probe_max_dim = []() {
    size_t longest = 1;
    for (auto dim : svs::distance::supported_dim_list) {
        if (dim != svs::Dynamic && dim > longest) {
            longest = dim;
        }
    }
    return longest;
}();

// A length for the svs::Dynamic kernels. Deliberately not a multiple of any
// vector width, so the epilogue is exercised too.
constexpr size_t probe_dynamic_dim = 97;

// One buffer serves every call, whichever extents the surface happens to declare.
constexpr size_t probe_buffer_dim = std::max(probe_max_dim, probe_dynamic_dim);

template <typename E> const E* buffer() {
    static const std::array<E, probe_buffer_dim> values = []() {
        std::array<E, probe_buffer_dim> filled{};
        filled.fill(static_cast<E>(1.0F));
        return filled;
    }();
    return values.data();
}

// svs::lib::MaybeStatic<svs::Dynamic> has no default constructor: a dynamic
// extent must be told its length.
template <size_t N> svs::lib::MaybeStatic<N> probe_length() {
    if constexpr (N == svs::Dynamic) {
        return svs::lib::MaybeStatic<N>(probe_dynamic_dim);
    } else {
        return svs::lib::MaybeStatic<N>();
    }
}

// Named directly rather than through L2::compute, which also reaches
// AVX_AVAILABILITY::NONE -- not in the surface, so every consumer instantiates it.
#define SVS_PROBE_ONE(Ea, Eb, N, LEVEL)                                                   \
    total += svs::distance::L2Impl<N, Ea, Eb, AVX_AVAILABILITY::LEVEL>::compute(          \
        buffer<Ea>(), buffer<Eb>(), probe_length<N>()                                     \
    );                                                                                    \
    total += svs::distance::IPImpl<N, Ea, Eb, AVX_AVAILABILITY::LEVEL>::compute(          \
        buffer<Ea>(), buffer<Eb>(), probe_length<N>()                                     \
    );                                                                                    \
    total +=                                                                              \
        svs::distance::CosineSimilarityImpl<N, Ea, Eb, AVX_AVAILABILITY::LEVEL>::compute( \
            buffer<Ea>(), buffer<Eb>(), 1.0F, probe_length<N>()                           \
        );

// The level's own type-pair list, not the full one: naming a pair the level has no
// kernel for would instantiate the generic template right here.
#define SVS_PROBE_TARGET(N, LEVEL)                         \
    if (host_satisfies<AVX_AVAILABILITY::LEVEL>()) {       \
        SVS_TYPE_PAIRS_FOR(LEVEL, SVS_PROBE_ONE, N, LEVEL) \
    }

float probe_all() {
    float total = 0;
    SVS_FOR_EACH_DISPATCH_TARGET(SVS_PROBE_TARGET)
    return total;
}

#undef SVS_PROBE_TARGET
#undef SVS_PROBE_ONE

} // namespace

int main() {
    // Printing keeps the calls above from being optimized away, and makes the run
    // a smoke test of every kernel this host can reach.
    std::printf("dispatch surface probe: %f\n", static_cast<double>(probe_all()));
    return 0;
}
