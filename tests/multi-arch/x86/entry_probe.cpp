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

// A consumer that reaches the kernels the way real code does: through the
// distance entry points, which name every ISA level unconditionally and pick one
// at runtime. Complements link_probe.cpp, which names the kernels directly and so
// checks the surface without depending on the entry points routing to it.

#include "svs/core/distance/cosine.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/inner_product.h"
#include "svs/multi-arch/x86/preprocessor.h"

#include "host_levels.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>

namespace {

// Deliberately not a multiple of any vector width, so the epilogue is exercised.
constexpr size_t entry_dynamic_dim = 97;

constexpr size_t fixed_extent(bool longest) {
    size_t result = longest ? 1 : svs::Dynamic;
    for (auto dim : svs::distance::supported_dim_list) {
        if (dim == svs::Dynamic) {
            continue;
        }
        result = longest ? std::max(result, dim) : std::min(result, dim);
    }
    return result;
}

// The extent the execution check breaks on: the surface's shortest fixed one, so
// the kernel it enters is a fully unrolled specialization rather than the epilogue.
constexpr size_t entry_report_dim = fixed_extent(false);

constexpr size_t entry_buffer_dim = std::max(fixed_extent(true), entry_dynamic_dim);

template <typename E> const E* buffer() {
    static const std::array<E, entry_buffer_dim> values = []() {
        std::array<E, entry_buffer_dim> filled{};
        filled.fill(static_cast<E>(1.0F));
        return filled;
    }();
    return values.data();
}

// A template rather than a macro body: the svs::Dynamic entry points take a
// length, and `if constexpr` only discards the other branch inside a template.
template <size_t N, typename Ea, typename Eb> float entry_one() {
    const Ea* a = buffer<Ea>();
    const Eb* b = buffer<Eb>();
    if constexpr (N == svs::Dynamic) {
        return svs::distance::L2::compute(a, b, entry_dynamic_dim) +
               svs::distance::IP::compute(a, b, entry_dynamic_dim) +
               svs::distance::CosineSimilarity::compute(a, b, 1.0F, entry_dynamic_dim);
    } else {
        return svs::distance::L2::compute<N>(a, b) + svs::distance::IP::compute<N>(a, b) +
               svs::distance::CosineSimilarity::compute<N>(a, b, 1.0F);
    }
}

#define SVS_ENTRY_ONE(Ea, Eb, N) total += entry_one<N, Ea, Eb>();
#define SVS_ENTRY_DIM(N) SVS_FOR_EACH_TYPE_PAIR(SVS_ENTRY_ONE, N)

float entry_all() {
    float total = 0;
    SVS_FOR_EACH_SUPPORTED_DIM(SVS_ENTRY_DIM)
    return total;
}

#undef SVS_ENTRY_DIM
#undef SVS_ENTRY_ONE

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--report") == 0) {
        // Read by cmake/dispatch-checks/check-dispatch-execution.cmake, which
        // breaks on this extent's kernels to see which level the call enters.
        std::printf("expect-level %d\n", svs_test::expected_level());
        std::printf("probe-extent %zu\n", entry_report_dim);
        std::printf(
            "one-call %f\n",
            static_cast<double>(entry_one<entry_report_dim, float, float>())
        );
        return 0;
    }

    // Printing keeps the calls from being optimized away, and makes the run a
    // smoke test of every entry point over the whole surface.
    std::printf("dispatch surface entry probe: %f\n", static_cast<double>(entry_all()));
    return 0;
}
