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
#include <string>
#include <variant>

// svs python bindings
#include "svs/python/allocator.h"

// pybind
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// svs
#include "svs/core/allocator.h"

namespace py = pybind11;

namespace svs::python::allocators {
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
} // namespace svs::python::allocators
