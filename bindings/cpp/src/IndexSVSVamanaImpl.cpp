/*
 * Copyright 2025 Intel Corporation
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

#include "IndexSVSVamanaImpl.h"
#include "IndexSVSImplUtils.h"

#include <algorithm>
#include <variant>

#include <svs/core/data.h>
#include <svs/core/distance.h>
#include <svs/extensions/vamana/scalar.h>
#include <svs/lib/float16.h>
#include <svs/orchestrators/dynamic_vamana.h>

namespace svs {
namespace runtime {
namespace {

std::variant<float, svs::Float16, std::int8_t>
get_storage_variant(IndexSVSVamanaImpl::StorageKind kind) {
    switch (kind) {
        case IndexSVSVamanaImpl::StorageKind::FP32:
            return float{};
        case IndexSVSVamanaImpl::StorageKind::FP16:
            return svs::Float16{};
        case IndexSVSVamanaImpl::StorageKind::SQI8:
            return std::int8_t{};
        default:
            throw ANNEXCEPTION("not supported SVS storage kind");
    }
}

svs::index::vamana::VamanaBuildParameters
get_build_parameters(const IndexSVSVamanaImpl::BuildParams& params) {
    return svs::index::vamana::VamanaBuildParameters{
        params.alpha,
        params.graph_max_degree,
        params.construction_window_size,
        params.max_candidate_pool_size,
        params.prune_to,
        params.use_full_search_history};
}

template <
    typename T,
    typename Alloc = svs::data::Blocked<svs::lib::Allocator<T>>,
    svs::data::ImmutableMemoryDataset Dataset,
    svs::threads::ThreadPool Pool>
    requires std::is_floating_point_v<T> || std::is_same_v<T, svs::Float16>
svs::data::SimpleData<T, svs::Dynamic, Alloc>
make_storage(const Dataset& data, Pool& pool) {
    svs::data::SimpleData<T, svs::Dynamic, Alloc> result(
        data.size(), data.dimensions(), Alloc{}
    );
    svs::threads::parallel_for(
        pool,
        svs::threads::StaticPartition(result.size()),
        [&](auto is, auto SVS_UNUSED(tid)) {
            for (auto i : is) {
                result.set_datum(i, data.get_datum(i));
            }
        }
    );
    return result;
}

template <
    typename T,
    typename Alloc = svs::data::Blocked<svs::lib::Allocator<T>>,
    svs::data::ImmutableMemoryDataset Dataset,
    svs::threads::ThreadPool Pool>
    requires std::is_integral_v<T>
svs::quantization::scalar::SQDataset<T, svs::Dynamic, Alloc>
make_storage(const Dataset& data, Pool& pool) {
    return svs::quantization::scalar::SQDataset<T, svs::Dynamic, Alloc>::compress(
        data, pool, Alloc{}
    );
}

template <typename ElementType>
svs::DynamicVamana*
init_impl_t(IndexSVSVamanaImpl* index, MetricType metric, size_t n, const float* x) {
    auto threadpool =
        svs::threads::ThreadPoolHandle(svs::threads::OMPThreadPool(omp_get_max_threads()));

    auto data = make_storage<ElementType>(
        svs::data::ConstSimpleDataView<float>(x, n, index->dim_), threadpool
    );

    std::vector<size_t> labels(data.size());
    std::iota(labels.begin(), labels.end(), 0);

    svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric));
    return distance_dispatcher([&](auto&& distance) {
        return new svs::DynamicVamana(svs::DynamicVamana::build<float>(
            get_build_parameters(index->build_params),
            std::move(data),
            std::move(labels),
            std::forward<decltype(distance)>(distance),
            std::move(threadpool)
        ));
    });
}

template <
    typename T,
    typename Alloc = svs::data::Blocked<svs::lib::Allocator<T>>,
    typename Enabler = void>
struct storage_type;

template <typename T, typename Alloc>
struct storage_type<
    T,
    Alloc,
    std::enable_if_t<std::is_floating_point_v<T> || std::is_same_v<T, svs::Float16>>> {
    using type = svs::data::SimpleData<T, svs::Dynamic, Alloc>;
};

template <typename T, typename Alloc>
struct storage_type<T, Alloc, std::enable_if_t<std::is_integral_v<T>>> {
    using type = svs::quantization::scalar::SQDataset<T, svs::Dynamic, Alloc>;
};

template <typename T, typename Alloc = svs::data::Blocked<svs::lib::Allocator<T>>>
using storage_type_t = typename storage_type<T, Alloc>::type;

template <typename ElementType>
svs::DynamicVamana* deserialize_impl_t(std::istream& stream, MetricType metric) {
    auto threadpool =
        svs::threads::ThreadPoolHandle(svs::threads::OMPThreadPool(omp_get_max_threads()));

    svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric));
    return distance_dispatcher([&](auto&& distance) {
        return new svs::DynamicVamana(
            svs::DynamicVamana::assemble<float, storage_type_t<ElementType>>(
                stream, std::forward<decltype(distance)>(distance), std::move(threadpool)
            )
        );
    });
}

svs::index::vamana::VamanaSearchParameters make_search_parameters(
    const std::unique_ptr<svs::DynamicVamana>& impl,
    const IndexSVSVamanaImpl::SearchParams& default_params,
    const IndexSVSVamanaImpl::SearchParams* params
) {
    if (!impl) {
        throw ANNEXCEPTION("Index not initialized");
    }

    auto search_window_size = default_params.search_window_size;
    auto search_buffer_capacity = default_params.search_buffer_capacity;

    if (params != nullptr) {
        if (params->search_window_size > 0)
            search_window_size = params->search_window_size;
        if (params->search_buffer_capacity > 0)
            search_buffer_capacity = params->search_buffer_capacity;
    }

    return impl->get_search_parameters().buffer_config(
        {search_window_size, search_buffer_capacity}
    );
}
} // namespace

IndexSVSVamanaImpl* IndexSVSVamanaImpl::build(
    size_t dim, MetricType metric, const BuildParams& params
) noexcept {
    try {
        auto index = new IndexSVSVamanaImpl(
            dim, params.graph_max_degree, metric, params.storage_kind
        );
        index->build_params = params;
        return index;
    } catch (...) { return nullptr; }
}

void IndexSVSVamanaImpl::destroy(IndexSVSVamanaImpl* impl) noexcept { delete impl; }

IndexSVSVamanaImpl::IndexSVSVamanaImpl() = default;

IndexSVSVamanaImpl::IndexSVSVamanaImpl(
    size_t d, size_t degree, MetricType metric, StorageKind storage
)
    : metric_type_(metric)
    , dim_(d)
    , build_params{
          storage,
          degree,
          degree < 4 ? degree : degree - 4,
          metric == MetricType::L2 ? 1.2f : 0.95f,
          40,
          200,
          true} {}

IndexSVSVamanaImpl::~IndexSVSVamanaImpl() = default;

Status IndexSVSVamanaImpl::add(size_t n, const float* x) noexcept {
    if (!impl) {
        return init_impl(n, x);
    }

    // construct sequential labels
    std::vector<size_t> labels(n);

    std::iota(labels.begin(), labels.end(), impl->size());

    auto data = svs::data::ConstSimpleDataView<float>(x, n, dim_);
    impl->add_points(data, labels);
    return Status_Ok;
}

void IndexSVSVamanaImpl::reset() noexcept {
    if (impl) {
        impl.reset();
    }
    ntotal_soft_deleted = 0;
}

Status IndexSVSVamanaImpl::search(
    size_t n,
    const float* x,
    size_t k,
    float* distances,
    size_t* labels,
    const SearchParams* params,
    IDFilter* filter
) const noexcept {
    if (!impl) {
        for (size_t i = 0; i < n; ++i) {
            distances[i] = std::numeric_limits<float>::infinity();
            labels[i] = -1;
        }
        return Status{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
    }

    if (k == 0) {
        return Status{ErrorCode::INVALID_ARGUMENT, "k must be greater than 0"};
    }

    auto sp = make_search_parameters(impl, default_search_params, params);

    // Simple search
    if (filter == nullptr) {
        auto queries = svs::data::ConstSimpleDataView<float>(x, n, dim_);

        // TODO: faiss use int64_t as label whereas SVS uses size_t?
        auto results = svs::QueryResultView<size_t>{
            svs::MatrixView<size_t>{
                svs::make_dims(n, k), static_cast<size_t*>(static_cast<void*>(labels))},
            svs::MatrixView<float>{svs::make_dims(n, k), distances}};
        impl->search(results, queries, sp);
        return Status_Ok;
    }

    // Selective search with IDSelector
    auto old_sp = impl->get_search_parameters();
    impl->set_search_parameters(sp);

    auto search_closure = [&](const auto& range, uint64_t SVS_UNUSED(tid)) {
        for (auto i : range) {
            // For every query
            auto query = std::span(x + i * dim_, dim_);
            auto curr_distances = std::span(distances + i * k, k);
            auto curr_labels = std::span(labels + i * k, k);

            auto iterator = impl->batch_iterator(query);
            size_t found = 0;
            do {
                iterator.next(k);
                for (auto& neighbor : iterator.results()) {
                    if (filter->is_member(neighbor.id())) {
                        curr_distances[found] = neighbor.distance();
                        curr_labels[found] = neighbor.id();
                        found++;
                        if (found == k) {
                            break;
                        }
                    }
                }
            } while (found < k && !iterator.done());
            // Pad with -1s
            for (; found < k; ++found) {
                curr_distances[found] = -1;
                curr_labels[found] = -1;
            }
        }
    };

    auto threadpool =
        svs::threads::OMPThreadPool(std::min(n, size_t(omp_get_max_threads())));

    svs::threads::parallel_for(
        threadpool, svs::threads::StaticPartition{n}, search_closure
    );

    impl->set_search_parameters(old_sp);

    return Status_Ok;
}

Status IndexSVSVamanaImpl::range_search(
    size_t n,
    const float* x,
    float radius,
    const ResultsAllocator& results,
    const SearchParams* params,
    IDFilter* filter
) const noexcept {
    if (!impl) {
        return Status{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
    }
    if (radius <= 0) {
        return Status{ErrorCode::INVALID_ARGUMENT, "radius must be greater than 0"};
    }

    auto sp = make_search_parameters(impl, default_search_params, params);
    auto old_sp = impl->get_search_parameters();
    impl->set_search_parameters(sp);

    // Using ResultHandler makes no sense due to it's complexity, overhead and
    // missed features; e.g. add_result() does not indicate whether result added
    // or not - we have to manually manage threshold comparison and id
    // selection.

    // Prepare output buffers
    std::vector<std::vector<svs::Neighbor<size_t>>> all_results(n);
    // Reserve space for allocation to avoid multiple reallocations
    // Use search_buffer_capacity as a heuristic
    const auto result_capacity = sp.buffer_config_.get_total_capacity();
    for (auto& res : all_results) {
        res.reserve(result_capacity);
    }

    svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric_type_));

    std::function<bool(float, float)> compare = distance_dispatcher([](auto&& dist) {
        return std::function<bool(float, float)>{svs::distance::comparator(dist)};
    });

    std::function<bool(size_t)> select = [](size_t) { return true; };
    if (filter != nullptr) {
        select = [&](size_t id) { return filter->is_member(id); };
    }

    // Set iterator batch size to search window size
    auto batch_size = sp.buffer_config_.get_search_window_size();

    auto range_search_closure = [&](const auto& range, uint64_t SVS_UNUSED(tid)) {
        for (auto i : range) {
            // For every query
            auto query = std::span(x + i * dim_, dim_);

            auto iterator = impl->batch_iterator(query);
            bool in_range = true;

            do {
                iterator.next(batch_size);
                for (auto& neighbor : iterator.results()) {
                    // SVS comparator functor returns true if the first distance
                    // is 'closer' than the second one
                    in_range = compare(neighbor.distance(), radius);
                    if (in_range) {
                        // Selective search with IDSelector
                        if (select(neighbor.id())) {
                            all_results[i].push_back(neighbor);
                        }
                    } else {
                        // Since iterator.results() are ordered by distance, we
                        // can stop processing
                        break;
                    }
                }
            } while (in_range && !iterator.done());
        }
    };

    auto threadpool =
        svs::threads::OMPThreadPool(std::min(n, size_t(omp_get_max_threads())));

    svs::threads::parallel_for(
        threadpool, svs::threads::StaticPartition{n}, range_search_closure
    );

    // Allocate output
    std::vector<size_t> result_counts(n);
    std::transform(
        all_results.begin(),
        all_results.end(),
        result_counts.begin(),
        [](const auto& res) { return res.size(); }
    );
    auto results_storage = results(result_counts);

    // Fill in results
    for (size_t q = 0, ofs = 0; q < n; ++q) {
        for (const auto& [id, distance] : all_results[q]) {
            results_storage.labels[ofs] = id;
            results_storage.distances[ofs] = distance;
            ofs++;
        }
    }

    impl->set_search_parameters(old_sp);
    return Status_Ok;
}

size_t IndexSVSVamanaImpl::remove_ids(const IDFilter& selector) noexcept {
    if (!impl) {
        return 0;
    }

    auto ids = impl->all_ids();
    std::vector<size_t> ids_to_delete;
    std::copy_if(ids.begin(), ids.end(), std::back_inserter(ids_to_delete), [&](size_t id) {
        return selector(id);
    });

    // SVS deletion is a soft deletion, meaning the corresponding vectors are
    // marked as deleted but still present in both the dataset and the graph,
    // and will be navigated through during search.
    // Actual cleanup happens once a large enough number of soft deleted vectors
    // are collected.
    impl->delete_points(ids_to_delete);
    // ntotal -= ids.size();
    ntotal_soft_deleted += ids_to_delete.size();

    auto ntotal = impl->size();
    const float cleanup_threshold = .5f;
    if (ntotal == 0 || (float)ntotal_soft_deleted / ntotal > cleanup_threshold) {
        impl->consolidate();
        impl->compact();
        ntotal_soft_deleted = 0;
    }
    return ids_to_delete.size();
}

Status IndexSVSVamanaImpl::init_impl(size_t n, const float* x) noexcept {
    if (impl) {
        return Status{ErrorCode::UNKNOWN_ERROR, "Index already initialized"};
    }

    impl.reset(std::visit(
        [&](auto element) {
            using ElementType = std::decay_t<decltype(element)>;
            return init_impl_t<ElementType>(this, metric_type_, n, x);
        },
        get_storage_variant(storage_kind)
    ));
    return Status_Ok;
}

Status IndexSVSVamanaImpl::serialize_impl(std::ostream& out) const noexcept {
    bool initialized = impl != nullptr;
    out.write(reinterpret_cast<const char*>(&initialized), sizeof(bool));

    if (initialized) {
        impl->save(out);
    }
    return Status_Ok;
}

Status IndexSVSVamanaImpl::deserialize_impl(std::istream& in) noexcept {
    if (impl) {
        return Status{
            ErrorCode::INVALID_ARGUMENT,
            "Cannot deserialize: SVS index already initialized."};
    }

    bool initialized = false;
    in.read(reinterpret_cast<char*>(&initialized), sizeof(bool));
    if (!initialized) {
        return Status_Ok;
    }

    impl.reset(std::visit(
        [&](auto element) {
            using ElementType = std::decay_t<decltype(element)>;
            return deserialize_impl_t<ElementType>(in, metric_type_);
        },
        get_storage_variant(storage_kind)
    ));
    return Status_Ok;
}

} // namespace runtime
} // namespace svs
