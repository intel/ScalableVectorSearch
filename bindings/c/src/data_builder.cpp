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

#include "data_builder.hpp"

#include "storage.hpp"

#include <svs/core/data/simple.h>
#include <svs/lib/dispatcher.h>
#include <svs/lib/misc.h>

#include <variant>

namespace svs::c_runtime {
// namespace {
template <typename DataBuilder>
size_t
estimate_size(DataBuilder builder, size_t num_vectors, size_t dimension, svs::lib::Empty) {
    using allocator_type = typename DataBuilder::allocator_type;
    static_assert(
        !svs::data::is_blocked_v<allocator_type>,
        "estimate_size requires a non-blocked allocator type."
    );
    return builder.estimate_size(num_vectors, dimension, allocator_type{});
}

template <typename DataBuilder>
size_t estimate_blocked_size(
    DataBuilder builder,
    size_t num_vectors,
    size_t dimension,
    svs::data::BlockingParameters block_params
) {
    using allocator_type = typename DataBuilder::allocator_type;
    static_assert(
        svs::data::is_blocked_v<allocator_type>,
        "estimate_blocked_size requires a blocked allocator type."
    );
    auto allocator = allocator_type{block_params};
    return builder.estimate_size(num_vectors, dimension, allocator);
}

template <typename Dispatcher>
void register_data_size_specializations(Dispatcher& dispatcher) {
    auto size_closure = [&dispatcher]<typename DataBuilder, typename = void>() {
        dispatcher.register_target(&estimate_size<DataBuilder>);
    };

    for_simple_specializations<false>(size_closure);
    for_leanvec_specializations<false>(size_closure);
    for_lvq_specializations<false>(size_closure);
    for_sq_specializations<false>(size_closure);

    auto blocked_size_closure = [&dispatcher]<typename DataBuilder, typename = void>() {
        dispatcher.register_target(&estimate_blocked_size<DataBuilder>);
    };

    for_simple_specializations<true>(blocked_size_closure);
    for_leanvec_specializations<true>(blocked_size_closure);
    for_lvq_specializations<true>(blocked_size_closure);
    for_sq_specializations<true>(blocked_size_closure);
}

using BlocksizeArg = std::variant<svs::lib::Empty, svs::data::BlockingParameters>;

using EstimateSizeDispatcher =
    svs::lib::Dispatcher<size_t, const Storage*, size_t, size_t, BlocksizeArg>;

const EstimateSizeDispatcher& build_data_size_dispatcher() {
    static EstimateSizeDispatcher dispatcher = [] {
        EstimateSizeDispatcher d{};
        register_data_size_specializations(d);
        return d;
    }();
    return dispatcher;
}

size_t dispatch_data_size_estimation(
    const Storage* storage, size_t num_vectors, size_t dimension, BlocksizeArg block_params
) {
    return build_data_size_dispatcher().invoke(
        storage, num_vectors, dimension, std::move(block_params)
    );
}
//} // namespace

size_t estimate_data_size(const Storage* storage, size_t num_vectors, size_t dimension) {
    if (storage == nullptr) {
        throw std::invalid_argument("Storage pointer cannot be null.");
    }
    if (num_vectors == 0) {
        throw std::invalid_argument("Number of vectors must be greater than zero.");
    }
    if (dimension == 0) {
        throw std::invalid_argument("Dimension must be greater than zero.");
    }
    return dispatch_data_size_estimation(
        storage, num_vectors, dimension, svs::lib::Empty{}
    );
}

size_t estimate_data_size_blocked(
    const Storage* storage,
    size_t num_vectors,
    size_t dimension,
    const svs::data::BlockingParameters& block_params
) {
    if (storage == nullptr) {
        throw std::invalid_argument("Storage pointer cannot be null.");
    }
    if (num_vectors == 0) {
        throw std::invalid_argument("Number of vectors must be greater than zero.");
    }
    if (dimension == 0) {
        throw std::invalid_argument("Dimension must be greater than zero.");
    }
    return dispatch_data_size_estimation(storage, num_vectors, dimension, block_params);
}
} // namespace svs::c_runtime
