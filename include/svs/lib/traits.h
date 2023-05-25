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

#include <concepts>

namespace svs::lib {

///
/// @brief Loader tags disambiguate the semantic behavior of an object loader.
///
struct AbstractLoaderTag {};

///
/// Tag for object reloader.
///
template <typename T> struct LoaderTraits {
    using type = typename T::loader_tag;
};

template <typename T> using loader_tag_t = typename LoaderTraits<T>::type;
template <typename T> inline constexpr loader_tag_t<T> loader_tag{};

} // namespace svs::lib
