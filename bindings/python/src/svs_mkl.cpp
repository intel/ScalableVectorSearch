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

// pysvs
#include "pysvs/svs_mkl.h"

// svs
#include "svs/lib/preprocessor.h"

// stl
#include <optional>

SVS_VALIDATE_BOOL_ENV(SVS_HAVE_MKL);
#if not SVS_HAVE_MKL

namespace pysvs {
bool have_mkl() { return false; }
std::optional<size_t> mkl_num_threads() { return std::nullopt; }
} // namespace pysvs

#else

#include <mkl.h>

namespace pysvs {
bool have_mkl() { return true; }
std::optional<size_t> mkl_num_threads() { return mkl_get_max_threads(); }
} // namespace pysvs

#endif
