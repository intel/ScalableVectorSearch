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

// svs python bindings
#include "svs/python/conversion.h"
#include "svs/python/common.h"
#include "svs/python/core.h"

// svs
#include "svs/quantization/lvq/lvq.h"

// pybind
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl/filesystem.h"

// stl
#include <filesystem>
#include <optional>

namespace lvq = svs::quantization::lvq;
namespace py = pybind11;

namespace svs::python {
namespace {

template <typename F> void register_specializations(F&& f) {
    // Pattern: Primary, Residual, Strategy
    f.template operator()<4, 0, lvq::Sequential>();
    f.template operator()<8, 0, lvq::Sequential>();
    f.template operator()<4, 4, lvq::Sequential>();
    f.template operator()<4, 8, lvq::Sequential>();
    f.template operator()<8, 8, lvq::Sequential>();
}

template <size_t Primary, size_t Residual, lvq::LVQPackingStrategy Strategy>
void compress(
    lvq::LVQLoader<Primary, Residual, svs::Dynamic, Strategy, Allocator> SVS_UNUSED(dispatch
    ),
    const std::filesystem::path& data_path,
    const std::filesystem::path& centroid_path,
    const std::filesystem::path& assignment_path,
    const std::filesystem::path& save_path,
    size_t num_threads
) {
    using dataset_t =
        svs::quantization::lvq::LVQDataset<Primary, Residual, svs::Dynamic, Strategy>;

    auto data = svs::VectorDataLoader<float>(data_path).load();
    auto centroids = svs::VectorDataLoader<float>(centroid_path).load();

    auto assignments = std::vector<uint64_t>(data.size());
    {
        auto stream = svs::lib::open_read(assignment_path);
        svs::lib::read_binary(stream, assignments);
    }

    // Allocate the storage dataset and set copy over the centroids.
    auto dst = dataset_t(data.size(), svs::lib::MaybeStatic(data.dimensions()));
    dst.reproducibility_set_centroids(centroids.cview());

    // Compress the dataset into the compressed destination.
    auto pool = svs::threads::NativeThreadPool(num_threads);
    svs::threads::run(
        pool,
        svs::threads::StaticPartition(data.size()),
        [&](auto is, auto SVS_UNUSED(tid)) {
            for (auto i : is) {
                dst.set_datum(i, data.get_datum(i), assignments.at(i));
            }
        }
    );

    // Save the result.
    svs::lib::save_to_disk(dst, save_path);
}

struct Compress {
    void operator()(
        const LVQ& source,
        const std::filesystem::path& data_path,
        const std::filesystem::path& centroid_path,
        const std::filesystem::path& assignment_path,
        const std::filesystem::path& save_path,
        size_t num_threads
    ) {
        auto dispatcher = svs::lib::Dispatcher<
            void,
            LVQ,
            const std::filesystem::path&,
            const std::filesystem::path&,
            const std::filesystem::path&,
            const std::filesystem::path&,
            size_t>();

        register_specializations([&]<size_t Primary, size_t Residual, typename Strategy>() {
            dispatcher.register_target(&compress<Primary, Residual, Strategy>);
        });

        dispatcher.invoke(
            source, data_path, centroid_path, assignment_path, save_path, num_threads
        );
    }
};

template <size_t Primary, size_t Residual, typename Strategy>
void decompress(
    lvq::LVQLoader<Primary, Residual, svs::Dynamic, Strategy, Allocator> loader,
    const std::filesystem::path& save_path
) {
    auto dataset = loader.load();
    auto dst = svs::data::SimpleData<float>(dataset.size(), dataset.dimensions());

    auto decompressor = dataset.decompressor();
    for (size_t i = 0, imax = dataset.size(); i < imax; ++i) {
        dst.set_datum(i, decompressor(dataset.get_datum(i)));
    }
    svs::lib::save_to_disk(dst, save_path);
}

struct Decompress {
    void operator()(const LVQ& loader, const std::filesystem::path& save_path) {
        auto dispatcher = svs::lib::Dispatcher<void, LVQ, const std::filesystem::path&>();
        register_specializations([&]<size_t Primary, size_t Residual, typename Strategy>() {
            dispatcher.register_target(&decompress<Primary, Residual, Strategy>);
        });
        dispatcher.invoke(loader, save_path);
    }
};

} // namespace

namespace conversion {

void wrap(py::module& m) {
    auto sub = m.def_submodule(
        "reproducibility", "Compatibility methods to reproduce paper results."
    );

    sub.def(
        "compress",
        Compress(),
        py::arg("source"),
        py::arg("data_path"),
        py::arg("centroid_path"),
        py::arg("assignment_path"),
        py::arg("save_path"),
        py::arg("num_threads") = 1
    );

    sub.def("decompress", Decompress(), py::arg("source"), py::arg("save_path"));
}

} // namespace conversion
} // namespace svs::python
