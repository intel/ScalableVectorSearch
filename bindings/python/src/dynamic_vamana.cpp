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

// project local
#include "dynamic_vamana.h"
#include "common.h"
#include "core.h"
#include "manager.h"
#include "vamana.h"

// svs
#include "svs/extensions/vamana/lvq.h"
#include "svs/lib/dispatcher.h"
#include "svs/orchestrators/dynamic_vamana.h"

// pybind
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// stl
#include <span>

/////
///// DynamicVamana
/////

namespace py = pybind11;
namespace dynamic_vamana {

namespace {

template <typename ElementType>
svs::DynamicVamana build_from_array(
    const svs::index::vamana::VamanaBuildParameters& parameters,
    py_contiguous_array_t<ElementType> py_data,
    py_contiguous_array_t<size_t> py_ids,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    auto dispatcher = svs::DistanceDispatcher(distance_type);
    return dispatcher([&](auto distance) {
        return svs::DynamicVamana::build<ElementType>(
            parameters,
            create_blocked_data(py_data),
            std::span(py_ids.data(), py_ids.size()),
            distance,
            num_threads
        );
    });
}

const char* BUILD_FROM_ARRAY_DOC = R"(
Construct a Vamana index over the given data, returning a searchable index.

Args:
    data: The dataset to index. **NOTE**: PySVS will maintain an internal copy of the
        dataset. This may change in future releases.
    parameters: Parameters controlling graph construction.
        See below for the documentation of this class.
    distance_type: The distance type to use for this dataset.
)";

template <typename ElementType>
void add_build_specialization(py::class_<svs::DynamicVamana>& index) {
    index.def_static(
        "build",
        &build_from_array<ElementType>,
        py::arg("parameters"),
        py::arg("data"),
        py::arg("ids"),
        py::arg("distance_type"),
        py::arg("num_threads"),
        BUILD_FROM_ARRAY_DOC
    );
}

template <typename ElementType>
void add_points(
    svs::DynamicVamana& index,
    const py_contiguous_array_t<ElementType>& py_data,
    const py_contiguous_array_t<size_t>& ids
) {
    if (py_data.ndim() != 2) {
        throw ANNEXCEPTION("Expected points to have 2 dimensions!");
    }
    if (ids.ndim() != 1) {
        throw ANNEXCEPTION("Expected ids to have 1 dimension!");
    }
    if (py_data.shape(0) != ids.shape(0)) {
        throw ANNEXCEPTION(
            "Expected IDs to be the same length as the number of rows in points!"
        );
    }
    index.add_points(data_view(py_data), std::span(ids.data(), ids.size()));
}

const char* ADD_POINTS_DOCSTRING = R"(
Add every point in ``points`` to the index, assigning the element-wise corresponding ID to
each point.

Args:
    points: A matrix of data whose rows, corresponding to points in R^n, will be added to
        the index.
    ids: Vector of ids to assign to each row in ``points``. Must have the same number of
        elements as ``points`` has rows.

Furthermore, all entries in ``ids`` must be unique and not already exist in the index.
If either of these does not hold, an exception will be thrown without mutating the
underlying index.
)";

template <typename ElementType>
void add_points_specialization(py::class_<svs::DynamicVamana>& index) {
    index.def(
        "add",
        &add_points<ElementType>,
        py::arg("points"),
        py::arg("ids"),
        ADD_POINTS_DOCSTRING
    );
}

///// Docstrings
// Put docstrings heere to hopefully make the implementation of `wrap` a bit less
// cluttered.
const char* CONSOLIDATE_DOCSTRING = R"(
Remove and patch around all deleted entries in the graph.
Should be called after a sufficient number of deletions to avoid the memory consumption of
the index monotonically increasing.
)";

const char* COMPACT_DOCSTRING = R"(
Remove any holes created in the graph and data by renumbering internal IDs.
Shrink the underlying data structures.
Following ``consolidate``, this can potentialy reduce the memory footprint of the index
if a sufficient number of points were deleted.
)";

const char* DELETE_DOCSTRING = R"(
Soft delete the IDs from the index. Soft deletion does not remove the IDs from the graph,
but prevents them from being returned from future searches.

Args:
    ids: The IDs to delete.

Each element in IDs must be unique and must correspond to a valid ID stored in the index.
Otherwise, an exception will be thrown. If an exception is thrown for this reason, the
index will be left unchanged from before the function call.
)";

const char* ALL_IDS_DOCSTRING = R"(
Return a Numpy vector of all IDs currently in the index.
)";

// Index saving.
void save_index(
    svs::DynamicVamana& index,
    const std::string& config_path,
    const std::string& graph_dir,
    const std::string& data_dir
) {
    index.save(config_path, graph_dir, data_dir);
}

// Assembly.
struct StandardAssemble_ {
    /// Keys:
    /// (0) - The type of the elements of the query vectors.
    /// (1) - The type of the elements of the data vectors.
    /// (2) - The requested distance type.
    /// (3) - Compile-time dimensionality
    using key_type = std::tuple<svs::DataType, svs::DataType, svs::DistanceType, size_t>;
    using mapped_type = std::function<svs::DynamicVamana(
        const std::filesystem::path& /*config_path*/,
        const UnspecializedGraphLoader& /*graph_loader*/,
        const UnspecializedVectorDataLoader& /*data_loader*/,
        size_t /*num_threads*/,
        bool /*debug_load_from_static*/
    )>;

    template <typename Q, typename T, typename Dist, size_t N>
    static std::pair<key_type, mapped_type>
    specialize(Type<Q> query_type, Type<T> data_type, Dist distance, Val<N> ndims) {
        key_type key = {
            unwrap(query_type),
            unwrap(data_type),
            svs::distance_type_v<Dist>,
            unwrap(ndims)};
        mapped_type fn = [=](const std::filesystem::path& config_path,
                             const UnspecializedGraphLoader& graph_loader,
                             const UnspecializedVectorDataLoader& data,
                             size_t num_threads,
                             bool debug_load_from_static) {
            auto load_graph = svs::lib::Lazy([&]() {
                return svs::graphs::SimpleBlockedGraph<uint32_t>::load(graph_loader.path());
            });

            auto load_data = svs::lib::Lazy([&]() {
                // Forward the allocator we wish to use
                using A = RebindAllocator<T>;
                return svs::data::BlockedData<T, N, A>::load(
                    data.path_, as_blocked(A(data.allocator_))
                );
            });

            return svs::DynamicVamana::assemble<Q>(
                config_path,
                load_graph,
                load_data,
                distance,
                num_threads,
                debug_load_from_static
            );
        };
        return std::make_pair(key, std::move(fn));
    }

    template <typename F> static void fill(F&& f) {
        for_standard_specializations(
            [&f](auto query_type, auto data_type, auto distance, auto ndims) {
                f(specialize(query_type, data_type, distance, ndims));
            }
        );
    }
};
using StandardAssembler = svs::lib::Dispatcher<StandardAssemble_>;

template <typename Loader> struct LVQAssemble_ {
    /// Keys:
    /// (0) - The requested distance type.
    /// (1) - Compile-time dimensionality
    using key_type = std::tuple<svs::DistanceType, size_t>;
    using mapped_type = std::function<svs::DynamicVamana(
        const std::filesystem::path& /*config_path*/,
        const UnspecializedGraphLoader& /*graph_loader*/,
        const Loader& /*data_loader*/,
        size_t /*num_threads*/,
        bool /*debug_load_from_static*/
    )>;

    template <typename Dist, size_t N>
    static std::pair<key_type, mapped_type> specialize(Dist distance, Val<N> ndims) {
        key_type key = {svs::distance_type_v<Dist>, unwrap(ndims)};
        mapped_type fn = [=](const std::filesystem::path& config_path,
                             const UnspecializedGraphLoader& graph_loader,
                             const Loader& data_loader,
                             size_t num_threads,
                             bool debug_load_from_static) {
            auto load_graph = svs::lib::Lazy([&]() {
                return svs::graphs::SimpleBlockedGraph<uint32_t>::load(graph_loader.path());
            });

            return svs::DynamicVamana::assemble<float>(
                config_path,
                load_graph,
                data_loader.refine(ndims, as_blocked),
                distance,
                num_threads,
                debug_load_from_static
            );
        };
        return std::make_pair(key, std::move(fn));
    }

    template <typename F> static void fill(F&& f) {
        for_compressed_specializations([&f](auto distance, auto ndims) {
            f(specialize(distance, ndims));
        });
    }
};
template <typename Loader> using LVQAssembler = svs::lib::Dispatcher<LVQAssemble_<Loader>>;

using DynamicVamanaAssembleTypes =
    std::variant<UnspecializedVectorDataLoader, LVQ8, LVQ4x8>;

svs::DynamicVamana assemble(
    const std::string& config_path,
    const UnspecializedGraphLoader& graph_loader,
    const DynamicVamanaAssembleTypes& data_loader,
    svs::DistanceType distance_type,
    svs::DataType query_type,
    bool enforce_dims,
    size_t num_threads,
    bool debug_load_from_static
) {
    return std::visit<svs::DynamicVamana>(
        [&](auto&& loader) {
            using T = std::decay_t<decltype(loader)>;
            if constexpr (std::is_same_v<T, UnspecializedVectorDataLoader>) {
                const auto& f = StandardAssembler::lookup(
                    !enforce_dims, loader.dims_, query_type, loader.type_, distance_type
                );
                return f(
                    config_path, graph_loader, loader, num_threads, debug_load_from_static
                );
            } else {
                const auto& f =
                    LVQAssembler<T>::lookup(!enforce_dims, loader.dims_, distance_type);
                return f(
                    config_path, graph_loader, loader, num_threads, debug_load_from_static
                );
            }
        },
        data_loader
    );
}

} // namespace

void wrap(py::module& m) {
    std::string name = "DynamicVamana";
    py::class_<svs::DynamicVamana> vamana(
        m, name.c_str(), "Top level class for the dynamic Vamana graph index."
    );

    add_search_specialization<float>(vamana);
    add_threading_interface(vamana);
    add_data_interface(vamana);

    // Vamana specific extentions.
    vamana::add_interface(vamana);

    // Dynamic interface.
    vamana.def_property(
        "alpha",
        &svs::DynamicVamana::get_alpha,
        &svs::DynamicVamana::set_alpha,
        "Read/Write (float): Get/set the alpha value used when adding and deleting points."
    );

    vamana.def_property(
        "construction_window_size",
        &svs::DynamicVamana::get_construction_window_size,
        &svs::DynamicVamana::set_construction_window_size,
        "Read/Write (int): Get/set the window size used when adding and deleting points."
    );

    vamana.def("consolidate", &svs::DynamicVamana::consolidate, CONSOLIDATE_DOCSTRING);
    vamana.def("compact", &svs::DynamicVamana::compact, COMPACT_DOCSTRING);

    // Reloading
    vamana.def(
        py::init(&assemble),
        py::arg("config_path"),
        py::arg("graph_loader"),
        py::arg("data_loader"),
        py::arg("distance") = svs::L2,
        py::arg("query_type") = svs::DataType::float32,
        py::arg("enforce_dims") = false,
        py::arg("num_threads") = 1,
        py::arg("debug_load_from_static") = false
    );

    // Index building.
    add_build_specialization<float>(vamana);

    // Index modification.
    add_points_specialization<float>(vamana);

    // Index Deletion.
    vamana.def(
        "delete",
        [](svs::DynamicVamana& index, const py_contiguous_array_t<size_t>& ids) {
            index.delete_points(as_span(ids));
        },
        py::arg("ids"),
        DELETE_DOCSTRING
    );

    // ID inspection
    vamana.def(
        "has_id",
        &svs::DynamicVamana::has_id,
        py::arg("id"),
        "Return whether the ID exists in the index."
    );

    vamana.def(
        "all_ids",
        [](const svs::DynamicVamana& index) {
            const auto& v = index.all_ids();
            // Populate a numpy-set
            auto npv = numpy_vector<size_t>(v.size());
            std::copy(v.begin(), v.end(), npv.mutable_unchecked().mutable_data());
            return npv;
        },
        ALL_IDS_DOCSTRING
    );

    // Saving
    vamana.def(
        "save",
        &save_index,
        py::arg("config_directory"),
        py::arg("graph_directory"),
        py::arg("data_directory"),
        R"(
Save a constructed index to disk (useful following index construction).

Args:
    config_directory: Directory where index configuration information will be saved.
    graph_directory: Directory where graph will be saved.
    data_directory: Directory where the dataset will be saved.


Note: All directories should be separate to avoid accidental name collision with any
auxiliary files that are needed when saving the various components of the index.

If the directory does not exist, it will be created if its parent exists.

It is the caller's responsibilty to ensure that no existing data will be
overwritten when saving the index to this directory.
    )"
    );
}

} // namespace dynamic_vamana
