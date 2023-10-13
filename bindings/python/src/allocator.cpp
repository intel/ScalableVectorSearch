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

#include <string>
#include <variant>

// pysvs
#include "allocator.h"

// pybind
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// svs
#include "svs/core/allocator.h"

namespace py = pybind11;

namespace allocators {
void wrap(pybind11::module& m) {
    // Hugepage Allocator
    // Use `std::byte` as a stand-in for various rebindings that will be used.
    py::class_<svs::HugepageAllocator<std::byte>> dram_allocator(
        m,
        "DRAM",
        R"(
Small class for an allocator capable of using huge pages. Prioritizes page use in the order:
1~GiB, 2~MiB, 4~KiB. See :ref:`hugepages` for more information on what huge pages are
and how to allocate them on your system.
        )"
    );
    dram_allocator.def(py::init<>(), "Construct an instance of the class.");
    dram_allocator.def("__str__", [](const py::object& /*arg*/) -> py::str {
        return py::str("DRAM()");
    });
    dram_allocator.def("__repr__", [](const py::object& /*arg*/) -> py::str {
        return py::str("DRAM()");
    });
}
} // namespace allocators
