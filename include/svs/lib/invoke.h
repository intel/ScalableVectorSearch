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
#pragma once

#include "svs/lib/preprocessor.h"

#include <type_traits>

namespace svs {

namespace func_ns {
struct dispatcher {
    template <typename Tag, typename... Args>
        requires requires(Tag&& tag, Args&&... args) {
                     svs_invoke(SVS_FWD(tag), SVS_FWD(args)...);
                 }
    SVS_FORCE_INLINE auto operator()(Tag&& tag, Args&&... args) const
        noexcept(noexcept(svs_invoke(SVS_FWD(tag), SVS_FWD(args)...)))
            -> decltype(svs_invoke(SVS_FWD(tag), SVS_FWD(args)...)) {
        return svs_invoke(SVS_FWD(tag), SVS_FWD(args)...);
    }
};
} // namespace func_ns

inline namespace callable_ns {
inline constexpr func_ns::dispatcher svs_invoke = {};
}

template <typename Tag, typename... Args>
concept svs_invocable = requires(Tag&& tag, Args&&... args) {
                            svs::svs_invoke(SVS_FWD(tag), SVS_FWD(args)...);
                        };

template <typename Tag, typename... Args>
using svs_invoke_result_t = std::invoke_result_t<decltype(svs::svs_invoke), Tag, Args...>;

template <auto& Tag> using tag_t = std::decay_t<decltype(Tag)>;

} // namespace svs
