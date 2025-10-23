/*
 * Copyright 2024 Intel Corporation
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

// svs python bindings
#include "svs/python/vamana_common.h"

// svs
#include "svs/index/vamana/index.h"
#include "svs/index/vamana/search_buffer.h"

// pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// fmt
#include "svs/third-party/fmt.h"

namespace py = pybind11;

namespace svs::python {
namespace {

std::string stringify_config(const svs::index::vamana::SearchBufferConfig& c) {
    return fmt::format(
        "SearchBufferConfig(search_window_size = {}, total_capacity = {})",
        c.get_search_window_size(),
        c.get_total_capacity()
    );
}

std::string stringify_search_params(const svs::index::vamana::VamanaSearchParameters& c) {
    auto fields = std::vector<std::string>(
        {fmt::format("buffer_config = {}", stringify_config(c.buffer_config_)),
         fmt::format("search_buffer_visited_set = {}", c.search_buffer_visited_set_),
         fmt::format("prefetch_lookahead = {}", c.prefetch_lookahead_),
         fmt::format("prefetch_step = {}", c.prefetch_step_)}
    );

    return fmt::format("VamanaSearchParameters({})", fmt::join(fields, ", "));
}

std::string stringify_calibration_params(const svs::index::vamana::CalibrationParameters& c
) {
    auto fields = std::vector<std::string>(
        {fmt::format("    search_window_size_upper = {}", c.search_window_size_upper_),
         fmt::format(
             "    search_window_capacity_upper = {}", c.search_window_capacity_upper_
         ),
         fmt::format("    timing_iterations = {}", c.timing_iterations_),
         fmt::format("    search_timeout = {}", c.search_timeout_),
         fmt::format("    prefetch_steps = [{}]", fmt::join(c.prefetch_steps_, ", ")),
         fmt::format("    search_buffer_optimization = {}", c.search_buffer_optimization_),
         fmt::format("    train_prefetchers = {}", c.train_prefetchers_),
         fmt::format(
             "    use_existing_parameter_values = {}", c.use_existing_parameter_values_
         )}
    );
    return fmt::format("VamanaCalibrationParameters(\n{}\n)", fmt::join(fields, ",\n"));
}

void wrap_search_buffer_config(py::module& m) {
    using SearchBufferConfig = svs::index::vamana::SearchBufferConfig;
    auto config = py::class_<SearchBufferConfig>{
        m,
        "SearchBufferConfig",
        R"(
Size configuration for the Vamana index search buffer.`
See also: :py:class:`svs.VamanaSearchParameters`,
:py:meth:`svs.Vamana.search_parameters`.

Attributes:
    search_window_size (int, read-only): The number of valid entries in the buffer
        that will be used to determine stopping conditions for graph search.
    search_buffer_capacity (int, read-only): The (expected) number of valid entries that
        will be available. Must be at least as large as `search_window_size`.
)"
    };

    config.def(py::init<>())
        .def(
            py::init<size_t>(),
            py::arg("search_window_size"),
            R"(
Configure with a given search window size. This constructor implicitly defaults
``search_buffer_capacity`` to ``search_window_size``.)"
        )
        .def(
            py::init<size_t, size_t>(),
            py::arg("search_window_size"),
            py::arg("search_buffer_capacity"),
            R"(
Configure with a given search window size and capacity.
Raises ``svs.ANNException`` if ``search_buffer_capacity < search_window_size``.)"
        )
        .def_property_readonly(
            "search_window_size", &SearchBufferConfig::get_search_window_size
        )
        .def_property_readonly(
            "search_buffer_capacity", &SearchBufferConfig::get_total_capacity
        )
        .def("__str__", &stringify_config)
        .def("__eq__", [](const SearchBufferConfig& x, const SearchBufferConfig& y) {
            return x == y;
        });
}

void wrap_search_parameters(py::module& m) {
    using VamanaSearchParameters = svs::index::vamana::VamanaSearchParameters;
    auto params = py::class_<VamanaSearchParameters>{
        m,
        "VamanaSearchParameters",
        R"(
Parameters controlling recall and performance of the VamanaIndex.
See also: :py:class:`Vamana.search_parameters`.

Attributes:
    buffer_config (:py:class:`svs.SearchBufferConfig`, read/write): Configuration state
        for the underlying search buffer.
    search_buffer_visited_set (bool, read/write): Enable/disable status of the search
        buffer visited set.
    prefetch_lookahead (unsigned int, read/write): The number of iterations ahead to
        prefetch during graph search.
    prefetch_step (unsigned int, read/write): The maximum number of iterations to prefetch
        at a time until the desired `prefetch_lookahead` is achieved. Setting this to 1
        is special and has the same effect setting this to `prefetch_lookahead + 1`.

Setting either ``prefetch_lookahead``  or ``prefetch_step`` to zero disables candidate
prefetching during search.
    )"
    };

    // N.B.: Keep defaults the same as the C++ class
    params
        .def(
            py::init<svs::index::vamana::SearchBufferConfig, bool, size_t, size_t>(),
            py::arg("buffer_config") = svs::index::vamana::SearchBufferConfig(),
            py::arg("search_buffer_visited_set") = false,
            py::arg("prefetch_lookahead") = 4,
            py::arg("prefetch_step") = 1
        )
        .def_readwrite("buffer_config", &VamanaSearchParameters::buffer_config_)
        .def_readwrite(
            "search_buffer_visited_set", &VamanaSearchParameters::search_buffer_visited_set_
        )
        .def_readwrite("prefetch_lookahead", &VamanaSearchParameters::prefetch_lookahead_)
        .def_readwrite("prefetch_step", &VamanaSearchParameters::prefetch_step_)
        .def("__str__", &stringify_search_params)
        .def(
            "__eq__",
            [](const VamanaSearchParameters& x, const VamanaSearchParameters& y) {
                return x == y;
            }
        );
}

void wrap_calibration_parameters(py::module& m) {
    using C = svs::index::vamana::CalibrationParameters;
    using SBO = C::SearchBufferOptimization;

    py::enum_<SBO>(
        m,
        "VamanaSearchBufferOptimization",
        "How should calibration optimize the search buffer."
    )
        .value("Disable", SBO::Disable, "Disable search buffer optimization.")
        .value(
            "All", SBO::All, "Optimize both search window size and capacity (if helpful)."
        )
        .value(
            "ROIOnly",
            SBO::ROIOnly,
            "Only optimize the search window size, setting the capacity equal to the "
            "search window size."
        )
        .value(
            "ROITuneUp",
            SBO::ROITuneUp,
            "Optimize the search buffer while keeping the capacity fixed. This routine can "
            "be used to slightly tweak accuracy numbers without relying on performance "
            "information."
        )
        .export_values();

    auto params = py::class_<C>{
        m,
        "VamanaCalibrationParameters",
        R"(
Hyper-parameters controlling performance tuning of the Vamana and DynamicVamana indexes.
See also: :py:meth:`Vamana.experimental_calibrate` and
:py:meth:`DynamicVamana.experimental_calibrate`.

Attributes:
    search_window_size_upper (int): The maximum search window size to check.
    search_window_capacity_upper (int): The maximum search capacity to check.
    timing_iterations (int): The maximum number iterations an indexed will be searched at a
        time for purposes of obtaining a measurement of search performance.
    search_timeout (float): A search bound (in seconds). Obtaining performance measurements
        will terminate early if the aggregate search time for a given setting exceeds this
        bound.
    prefetch_steps (List[int]): Steps to try when optimizing Prefetching.
    search_buffer_optimization (:py:class:`svs.VamanaSearchBufferOptimization`): Setting
        for optimizing the index search buffer.

        - Disable: Do not optimize the search buffer at all.
        - All: Optimize both search window size and capacity.
        - ROIOnly: Only optimize the search window size. Capacity will always be equal to
          the search window size.

    train_prefetchers (bool): Flag to train prefetch parameters.
    use_existing_parameter_values (bool): Should optimization use existing search parameters
        or should it use defaults instead.
)"
    };

    // N.B.: Keep defaults the same as the C++ class
    params.def(py::init<>(), "Instantiate with default parameters.")
        .def_readwrite("search_window_size_upper", &C::search_window_size_upper_)
        .def_readwrite("search_window_capacity_upper", &C::search_window_capacity_upper_)
        .def_readwrite("timing_iterations", &C::timing_iterations_)
        .def_readwrite("search_timeout", &C::search_timeout_)
        .def_readwrite("prefetch_steps", &C::prefetch_steps_)
        .def_readwrite("search_buffer_optimization", &C::search_buffer_optimization_)
        .def_readwrite("train_prefetchers", &C::train_prefetchers_)
        .def_readwrite("use_existing_parameter_values", &C::use_existing_parameter_values_)
        .def("__str__", &stringify_calibration_params)
        .def(
            "_repr_pretty_",
            [](const C& params, py::object& ipython_printer, bool SVS_UNUSED(cycle)) {
                ipython_printer.attr("text")(stringify_calibration_params(params));
            }
        );
}

} // namespace

namespace vamana {

void wrap_common(py::module& m) {
    wrap_search_buffer_config(m);
    wrap_search_parameters(m);
    wrap_calibration_parameters(m);
}

} // namespace vamana
} // namespace svs::python
