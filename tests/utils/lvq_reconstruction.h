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

#include "svs/core/data.h"

namespace svs_test {

/// @brief Check the quality of the LVQ reconstruction.
void check_lvq_reconstruction(
    svs::data::ConstSimpleDataView<float> original,
    svs::data::ConstSimpleDataView<float> reconstructed,
    size_t primary,
    size_t residual
);

} // namespace svs_test
