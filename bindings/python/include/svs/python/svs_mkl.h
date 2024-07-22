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

#include <optional>

namespace svs::python {

/// Return `true` if svs was build to link with MKL. Otherwise, return `false`.
bool have_mkl();

///
/// @brief Return the number of threads used by MKL.
///
/// If ``have_mkl()`` returns false, return an empty optional.
std::optional<size_t> mkl_num_threads();

} // namespace svs::python
