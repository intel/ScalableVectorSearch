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
