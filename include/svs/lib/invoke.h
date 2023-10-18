/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
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
