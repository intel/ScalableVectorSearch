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
    SVS_FORCE_INLINE constexpr auto operator()(Tag&& tag, Args&&... args) const
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
