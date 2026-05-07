/*
 * Copyright 2026 Intel Corporation
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

#pragma once

#include "ivf_index_impl.h"

namespace svs {
namespace runtime {

// Dynamic IVF index implementation (non-LeanVec storage kinds)
class DynamicIVFIndexImpl {
    using allocator_type = svs::data::Blocked<svs::lib::Allocator<float>>;

  public:
    DynamicIVFIndexImpl(
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        const IVFIndex::BuildParams& params,
        const IVFIndex::SearchParams& default_search_params,
        size_t num_threads,
        size_t intra_query_threads
    )
        : dim_{dim}
        , metric_type_{metric}
        , storage_kind_{storage_kind}
        , build_params_{params}
        , default_search_params_{default_search_params}
        , num_threads_{num_threads == 0 ? static_cast<size_t>(omp_get_max_threads()) : num_threads}
        , intra_query_threads_{intra_query_threads} {
        if (!ivf_storage::is_supported_non_leanvec_storage_kind(storage_kind)) {
            throw StatusException{
                ErrorCode::INVALID_ARGUMENT,
                "The specified storage kind is not compatible with DynamicIVFIndex. "
                "Use DynamicIVFIndexLeanVecImpl for LeanVec storage kinds."};
        }
    }

    size_t size() const { return impl_ ? impl_->size() : 0; }

    size_t dimensions() const { return dim_; }

    MetricType metric_type() const { return metric_type_; }

    StorageKind get_storage_kind() const { return storage_kind_; }

    void build(data::ConstSimpleDataView<float> data, std::span<const size_t> ids) {
        if (impl_) {
            throw StatusException{ErrorCode::RUNTIME_ERROR, "Index already initialized"};
        }
        init_impl(data, ids);
    }

    void
    add(data::ConstSimpleDataView<float> data,
        std::span<const size_t> ids,
        bool reuse_empty = false) {
        if (!impl_) {
            // First add initializes the index
            init_impl(data, ids);
            return;
        }
        impl_->add_points(data, ids, reuse_empty);
    }

    size_t remove(std::span<const size_t> ids) {
        if (!impl_) {
            throw StatusException{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
        }
        return impl_->delete_points(ids);
    }

    size_t remove_selected(const IDFilter& selector) {
        if (!impl_) {
            throw StatusException{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
        }

        auto ids = impl_->all_ids();
        std::vector<size_t> ids_to_delete;
        std::copy_if(
            ids.begin(),
            ids.end(),
            std::back_inserter(ids_to_delete),
            [&](size_t id) { return selector(id); }
        );

        return impl_->delete_points(ids_to_delete);
    }

    bool has_id(size_t id) const {
        if (!impl_) {
            return false;
        }
        return impl_->has_id(id);
    }

    void consolidate() {
        if (!impl_) {
            throw StatusException{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
        }
        impl_->consolidate();
    }

    void compact(size_t batchsize = 1'000'000) {
        if (!impl_) {
            throw StatusException{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
        }
        impl_->compact(batchsize);
    }

    void search(
        svs::QueryResultView<size_t> result,
        svs::data::ConstSimpleDataView<float> queries,
        const IVFIndex::SearchParams* params = nullptr,
        IDFilter* filter = nullptr
    ) const {
        if (!impl_) {
            auto& dists = result.distances();
            std::fill(dists.begin(), dists.end(), Unspecify<float>());
            auto& inds = result.indices();
            std::fill(inds.begin(), inds.end(), Unspecify<size_t>());
            throw StatusException{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
        }

        if (queries.size() == 0) {
            return;
        }

        const size_t k = result.n_neighbors();
        if (k == 0) {
            throw StatusException{ErrorCode::INVALID_ARGUMENT, "k must be greater than 0"};
        }

        auto sp = make_search_parameters(params);

        // Simple search
        if (filter == nullptr) {
            impl_->search(result, queries, sp);
            return;
        }

        // Selective search with IDFilter: use batch iterator to over-fetch and
        // filter per-neighbor, mirroring the Vamana approach.
        auto old_sp = impl_->get_search_parameters();
        auto sp_restore = svs::lib::make_scope_guard([&]() noexcept {
            impl_->set_search_parameters(old_sp);
        });
        impl_->set_search_parameters(sp);

        float filter_stop = 0.0f;
        bool filter_estimate_batch = true;
        if (params) {
            set_if_specified(filter_stop, params->filter_stop);
            set_if_specified(filter_estimate_batch, params->filter_estimate_batch);
        }
        const auto max_batch_size = impl_->size();

        // Pre-search filter sampling: estimate hit rate before cluster traversal.
        size_t sampled = 0;
        size_t sample_hits = 0;
        const auto initial_batch_hint = std::max(k, size_t{1});
        auto initial_batch_size = initial_batch_hint;
        if (filter_estimate_batch) {
            std::tie(sampled, sample_hits) = sample_filter_hits(
                *filter,
                max_batch_size,
                [this](size_t id) { return impl_->has_id(id); },
                sample_size_for_filter_stop(filter_stop)
            );
            if (should_stop_filtered_search(sampled, sample_hits, filter_stop)) {
                pad_empty_results(result, queries.size(), k);
                return;
            }
            initial_batch_size = predict_further_processing(
                sampled, sample_hits, k, initial_batch_hint, max_batch_size
            );
        }

        for (size_t i = 0; i < queries.size(); ++i) {
            auto query = queries.get_datum(i);
            auto iterator =
                impl_->batch_iterator(std::span<const float>(query.data(), query.size()));
            size_t found = 0;
            size_t total_checked = 0;
            auto batch_size = initial_batch_size;
            do {
                batch_size = predict_further_processing(
                    total_checked, found, k, batch_size, max_batch_size
                );
                iterator.next(batch_size);
                total_checked += iterator.size();
                for (const auto& neighbor : iterator.results()) {
                    if (filter->is_member(neighbor.id())) {
                        result.set(neighbor, i, found);
                        ++found;
                        if (found == k) {
                            break;
                        }
                    }
                }
                if (should_stop_filtered_search(total_checked, found, filter_stop)) {
                    found = 0;
                    break;
                }
            } while (found < k && !iterator.done());

            for (size_t j = found; j < k; ++j) {
                result.set(
                    svs::Neighbor<size_t>{Unspecify<size_t>(), Unspecify<float>()}, i, j
                );
            }
        }
    }

    void save(std::ostream& out) const {
        if (!impl_) {
            throw StatusException{
                ErrorCode::NOT_INITIALIZED,
                "Cannot serialize: DynamicIVF index not initialized."};
        }
        impl_->save(out);
    }

    void set_num_threads(size_t num_threads) {
        if (num_threads == 0) {
            num_threads = static_cast<size_t>(omp_get_max_threads());
        }
        num_threads_ = num_threads;
        if (impl_) {
            impl_->set_threadpool(svs::threads::DefaultThreadPool(num_threads));
        }
    }

    size_t get_num_threads() const {
        if (impl_) {
            return impl_->get_num_threads();
        }
        return num_threads_;
    }

    void set_intra_query_threads(size_t intra_query_threads) {
        if (intra_query_threads == 0) {
            throw StatusException{
                ErrorCode::INVALID_ARGUMENT, "intra_query_threads must be at least 1"};
        }
        intra_query_threads_ = intra_query_threads;
        if (impl_) {
            impl_->set_num_intra_query_threads(intra_query_threads);
        }
    }

    size_t get_intra_query_threads() const {
        if (impl_) {
            return impl_->get_num_intra_query_threads();
        }
        return intra_query_threads_;
    }

    static DynamicIVFIndexImpl* load(
        std::istream& in,
        MetricType metric,
        StorageKind storage_kind,
        size_t num_threads,
        size_t intra_query_threads
    ) {
        if (num_threads == 0) {
            num_threads = static_cast<size_t>(omp_get_max_threads());
        }

        // Dispatch on storage kind to load with correct data type
        return ivf_storage::dispatch_ivf_storage_kind<allocator_type>(
            storage_kind,
            [&](auto tag) {
                using Tag = decltype(tag);
                using DataType = typename Tag::type;

                svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric));
                return distance_dispatcher([&](auto&& distance) {
                    auto impl = std::make_unique<svs::DynamicIVF>(
                        svs::DynamicIVF::assemble<float, svs::BFloat16, DataType>(
                            in,
                            std::forward<decltype(distance)>(distance),
                            num_threads,
                            intra_query_threads
                        )
                    );
                    return new DynamicIVFIndexImpl(
                        std::move(impl),
                        metric,
                        storage_kind,
                        num_threads,
                        intra_query_threads
                    );
                });
            }
        );
    }

  protected:
    // Constructor used during loading
    DynamicIVFIndexImpl(
        std::unique_ptr<svs::DynamicIVF>&& impl,
        MetricType metric,
        StorageKind storage_kind,
        size_t num_threads,
        size_t intra_query_threads
    )
        : dim_{impl->dimensions()}
        , metric_type_{metric}
        , storage_kind_{storage_kind}
        , num_threads_{num_threads}
        , intra_query_threads_{intra_query_threads}
        , impl_{std::move(impl)} {
        // Extract default search params from loaded index
        auto loaded_params = impl_->get_search_parameters();
        default_search_params_.n_probes = loaded_params.n_probes_;
        default_search_params_.k_reorder = loaded_params.k_reorder_;
    }

    // Constructor used by subclasses (LeanVec) that handle their own validation
    DynamicIVFIndexImpl(
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        const IVFIndex::BuildParams& params,
        const IVFIndex::SearchParams& default_search_params,
        size_t num_threads,
        size_t intra_query_threads,
        bool /*skip_validation*/
    )
        : dim_{dim}
        , metric_type_{metric}
        , storage_kind_{storage_kind}
        , build_params_{params}
        , default_search_params_{default_search_params}
        , num_threads_{num_threads == 0 ? static_cast<size_t>(omp_get_max_threads()) : num_threads}
        , intra_query_threads_{intra_query_threads} {
        // Subclasses handle their own storage kind validation
    }

    svs::index::ivf::IVFBuildParameters ivf_build_parameters() const {
        svs::index::ivf::IVFBuildParameters result;
        set_if_specified(result.num_centroids_, build_params_.num_centroids);
        set_if_specified(result.minibatch_size_, build_params_.minibatch_size);
        set_if_specified(result.num_iterations_, build_params_.num_iterations);
        if (is_specified(build_params_.is_hierarchical)) {
            result.is_hierarchical_ = build_params_.is_hierarchical.is_enabled();
        }
        set_if_specified(result.training_fraction_, build_params_.training_fraction);
        set_if_specified(
            result.hierarchical_level1_clusters_, build_params_.hierarchical_level1_clusters
        );
        set_if_specified(result.seed_, build_params_.seed);
        return result;
    }

    svs::index::ivf::IVFSearchParameters
    make_search_parameters(const IVFIndex::SearchParams* params) const {
        // Start with default parameters
        svs::index::ivf::IVFSearchParameters result;
        if (is_specified(default_search_params_.n_probes)) {
            result.n_probes_ = default_search_params_.n_probes;
        }
        if (is_specified(default_search_params_.k_reorder)) {
            result.k_reorder_ = default_search_params_.k_reorder;
        }

        // Override with user-specified parameters
        if (params) {
            set_if_specified(result.n_probes_, params->n_probes);
            set_if_specified(result.k_reorder_, params->k_reorder);
        }

        return result;
    }

    void init_impl(data::ConstSimpleDataView<float> data, std::span<const size_t> ids) {
        auto build_params = ivf_build_parameters();
        auto threadpool =
            svs::threads::ThreadPoolHandle(svs::threads::DefaultThreadPool(num_threads_));

        svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric_type_));

        impl_.reset(distance_dispatcher([&](auto&& distance) {
            // Build clustering using BFloat16 for efficiency (AMX support)
            auto clustering = svs::IVF::build_clustering<svs::BFloat16>(
                build_params, data, std::forward<decltype(distance)>(distance), num_threads_
            );

            // Dispatch on storage kind to compress and assemble
            return ivf_storage::dispatch_ivf_storage_kind<allocator_type>(
                storage_kind_,
                [&](auto tag) {
                    // Compress data to target storage type using the factory
                    auto compressed_data =
                        ivf_storage::make_ivf_blocked_storage(tag, data, threadpool);

                    return new svs::DynamicIVF(
                        svs::DynamicIVF::assemble_from_clustering<float>(
                            std::move(clustering),
                            std::move(compressed_data),
                            ids,
                            std::forward<decltype(distance)>(distance),
                            num_threads_,
                            intra_query_threads_
                        )
                    );
                }
            );
        }));
    }

    // Data members
    size_t dim_;
    MetricType metric_type_;
    StorageKind storage_kind_;
    IVFIndex::BuildParams build_params_;
    IVFIndex::SearchParams default_search_params_;
    size_t num_threads_;
    size_t intra_query_threads_;
    std::unique_ptr<svs::DynamicIVF> impl_;
};

#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
// Dynamic IVF index implementation for LeanVec storage kinds
class DynamicIVFIndexLeanVecImpl : public DynamicIVFIndexImpl {
    using allocator_type = svs::data::Blocked<svs::lib::Allocator<std::byte>>;

  public:
    using LeanVecMatricesType = LeanVecTrainingDataImpl::LeanVecMatricesType;

    // Constructor for building with training data
    DynamicIVFIndexLeanVecImpl(
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        const LeanVecTrainingDataImpl& training_data,
        const IVFIndex::BuildParams& params,
        const IVFIndex::SearchParams& default_search_params,
        size_t num_threads,
        size_t intra_query_threads
    )
        : DynamicIVFIndexImpl{dim, metric, storage_kind, params, default_search_params, num_threads, intra_query_threads, /*skip_validation=*/true}
        , leanvec_dims_{training_data.get_leanvec_dims()}
        , leanvec_matrices_{training_data.get_leanvec_matrices()} {
        check_leanvec_storage_kind(storage_kind);
    }

    // Constructor for building without pre-computed matrices (will compute during build)
    DynamicIVFIndexLeanVecImpl(
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        size_t leanvec_dims,
        const IVFIndex::BuildParams& params,
        const IVFIndex::SearchParams& default_search_params,
        size_t num_threads,
        size_t intra_query_threads
    )
        : DynamicIVFIndexImpl{dim, metric, storage_kind, params, default_search_params, num_threads, intra_query_threads, /*skip_validation=*/true}
        , leanvec_dims_{leanvec_dims}
        , leanvec_matrices_{std::nullopt} {
        check_leanvec_storage_kind(storage_kind);
    }

    // Constructor for loading
    DynamicIVFIndexLeanVecImpl(
        std::unique_ptr<svs::DynamicIVF>&& impl,
        MetricType metric,
        StorageKind storage_kind,
        size_t num_threads,
        size_t intra_query_threads
    )
        : DynamicIVFIndexImpl{std::move(impl), metric, storage_kind, num_threads, intra_query_threads}
        , leanvec_dims_{0}
        , leanvec_matrices_{std::nullopt} {
        check_leanvec_storage_kind(storage_kind);
    }

    void build(data::ConstSimpleDataView<float> data, std::span<const size_t> ids) {
        if (impl_) {
            throw StatusException{ErrorCode::RUNTIME_ERROR, "Index already initialized"};
        }
        init_leanvec_impl(data, ids);
    }

    static DynamicIVFIndexLeanVecImpl* load(
        std::istream& in,
        MetricType metric,
        StorageKind storage_kind,
        size_t num_threads,
        size_t intra_query_threads
    ) {
        if (num_threads == 0) {
            num_threads = static_cast<size_t>(omp_get_max_threads());
        }

        return ivf_storage::dispatch_ivf_leanvec_storage_kind<allocator_type>(
            storage_kind,
            [&](auto tag) {
                using Tag = decltype(tag);
                using DataType = typename Tag::type;

                svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric));
                return distance_dispatcher([&](auto&& distance) {
                    auto impl = std::make_unique<svs::DynamicIVF>(
                        svs::DynamicIVF::assemble<float, svs::BFloat16, DataType>(
                            in,
                            std::forward<decltype(distance)>(distance),
                            num_threads,
                            intra_query_threads
                        )
                    );
                    return new DynamicIVFIndexLeanVecImpl(
                        std::move(impl),
                        metric,
                        storage_kind,
                        num_threads,
                        intra_query_threads
                    );
                });
            }
        );
    }

  protected:
    size_t leanvec_dims_;
    std::optional<LeanVecMatricesType> leanvec_matrices_;

    void check_leanvec_storage_kind(StorageKind kind) {
        if (!ivf_storage::is_leanvec_storage_kind(kind)) {
            throw StatusException(
                ErrorCode::INVALID_ARGUMENT, "LeanVec storage kind required"
            );
        }
        if (!svs::detail::lvq_leanvec_enabled()) {
            throw StatusException(
                ErrorCode::NOT_IMPLEMENTED,
                "LeanVec storage kind requested but not supported by CPU"
            );
        }
    }

    void
    init_leanvec_impl(data::ConstSimpleDataView<float> data, std::span<const size_t> ids) {
        auto build_params = ivf_build_parameters();
        auto threadpool =
            svs::threads::ThreadPoolHandle(svs::threads::DefaultThreadPool(num_threads_));

        svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric_type_));

        impl_.reset(distance_dispatcher([&](auto&& distance) {
            // Build clustering using BFloat16 for efficiency (AMX support)
            auto clustering = svs::IVF::build_clustering<svs::BFloat16>(
                build_params, data, std::forward<decltype(distance)>(distance), num_threads_
            );

            // Dispatch on LeanVec storage kind to compress and assemble
            return ivf_storage::dispatch_ivf_leanvec_storage_kind<allocator_type>(
                storage_kind_,
                [&](auto tag) {
                    // Compress data to LeanVec storage type using the factory with matrices
                    auto compressed_data = ivf_storage::make_ivf_blocked_leanvec_storage(
                        tag, data, threadpool, leanvec_dims_, leanvec_matrices_
                    );

                    return new svs::DynamicIVF(
                        svs::DynamicIVF::assemble_from_clustering<float>(
                            std::move(clustering),
                            std::move(compressed_data),
                            ids,
                            std::forward<decltype(distance)>(distance),
                            num_threads_,
                            intra_query_threads_
                        )
                    );
                }
            );
        }));
    }
};
#endif // SVS_RUNTIME_HAVE_LVQ_LEANVEC

} // namespace runtime
} // namespace svs
