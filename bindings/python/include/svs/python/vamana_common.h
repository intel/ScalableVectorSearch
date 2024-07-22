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

#include <pybind11/pybind11.h>

// Wraps the following data structures:
// svs::index::vamana::SearchBufferConfig
// svs::index::vamana::VamanaSearchParameters
namespace svs::python::vamana {
void wrap_common(pybind11::module& m);
}
