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

#pragma once
#include "svs/index/vamana/dynamic_index.h"
#include "svs/index/vamana/iterator.h"

namespace svs::index::vamana {

/// @brief A multi-vector batch iterator for retrieving neighbors with unique labels from
/// the index in batches.
///
/// This iterator abstracts the process of retrieving neighbors in fixed-size batches
/// while maintaining internal state for efficient graph traversal.
/// In multi-vector scenario,
/// each label can have multiple vectors.
/// This iterator ensures that neighbors are retrieved with unique labels
template <typename Index, typename QueryType> class MultiBatchIterator {
    using label_type = size_t;
    using external_id_type = size_t;
    using value_type = Neighbor<label_type>;

    // Private type aliases
    using result_buffer_type = std::vector<value_type>;

    using ParentIndex = typename Index::ParentIndex;
    using compare = typename Index::compare;

  public:
    /// Random-access iterator to `value_type` over the current batch of results.
    using iterator = typename result_buffer_type::iterator;
    /// Random-access iterator to `const value_type` over the current batch of results.
    using const_iterator = typename result_buffer_type::const_iterator;

    MultiBatchIterator(
        const Index& index,
        std::span<const QueryType> query,
        size_t extra_search_buffer_capacity = svs::UNSIGNED_INTEGER_PLACEHOLDER
    )
        : index_{index}
        , batch_iterator_{index.get_parent_index(), query, extra_search_buffer_capacity} {}

    void next(
        size_t batch_size,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) {
        const auto& external_to_label = index_.get_external_to_label_lookup();
        auto results_copy = results_;
        results_.clear();
        get_results_from_extra(batch_size);

        while (results_.size() < batch_size && !batch_iterator_.done()) {
            try {
                batch_iterator_.next(batch_size, cancel);
            } catch (const ANNException& error) {
                results_ = std::move(results_copy);
                throw error;
            }
            for (auto& result : batch_iterator_) {
                auto label = external_to_label.at(result.id());
                auto found_in_returned = returned_.find(label);
                auto new_result = Neighbor<label_type>{label, result.distance()};

                if (found_in_returned == returned_.end()) {
                    if (results_.size() < batch_size) {
                        returned_.insert(label);
                        results_.push_back(std::move(new_result));
                    } else {
                        extra_results_.push_back(std::move(new_result));
                    }
                } else {
                    // results_ should be small enough to use find
                    auto found_in_results = std::find_if(
                        results_.begin(),
                        results_.end(),
                        [label](const auto& res) { return res.id() == label; }
                    );
                    if (found_in_results != results_.end()) {
                        *found_in_results =
                            std::min(*found_in_results, new_result, TotalOrder(compare{}));
                    }
                }
            }
        }

        ++iteration_;
        return;
    }

    size_t batch_number() const { return iteration_; }

    void update(std::span<const QueryType> newquery) {
        iteration_ = 0;
        returned_.clear();
        results_.clear();
        extra_results_.clear();
        batch_iterator_.update(newquery);
    }

    iterator begin() { return results_.begin(); }
    iterator end() { return results_.end(); }
    const_iterator begin() const { return results_.begin(); }
    const_iterator end() const { return results_.end(); }
    const_iterator cbegin() const { return results_.cbegin(); }
    const_iterator cend() const { return results_.cend(); }
    size_t size() const { return results_.size(); }

    bool done() const { return batch_iterator_.done() && extra_results_.empty(); }

    std::span<const value_type> contents() const { return lib::as_const_span(results_); }

  private:
    void get_results_from_extra(size_t batch_size) {
        // sort to get the best candidate from the back
        std::sort(extra_results_.rbegin(), extra_results_.rend(), TotalOrder(compare{}));

        while (results_.size() < batch_size && !extra_results_.empty()) {
            auto best = extra_results_.back();
            extra_results_.pop_back();

            if (returned_.find(best.id()) == returned_.end()) {
                returned_.insert(best.id());
                results_.push_back(std::move(best));
            }
        }

        return;
    }

    const Index& index_;
    size_t iteration_ = 0;
    std::unordered_set<label_type> returned_;
    std::vector<Neighbor<label_type>> results_;
    std::vector<Neighbor<label_type>> extra_results_;
    BatchIterator<ParentIndex, QueryType> batch_iterator_;
};

template <graphs::MemoryGraph Graph, typename Data, typename Dist>
class MultiMutableVamanaIndex {
  public:
    static constexpr bool supports_insertions = true;
    static constexpr bool supports_deletions = true;
    static constexpr bool supports_saving = false; // temporary disable for now
    static constexpr bool needs_id_translation = true;

    using ParentIndex = MutableVamanaIndex<Graph, Data, Dist>;
    using compare = distance::compare_t<Dist>;
    using Idx = typename ParentIndex::Idx;
    using search_parameters_type = typename ParentIndex::search_parameters_type;
    using external_id_type = typename ParentIndex::external_id_type;
    using scratchspace_type = typename ParentIndex::scratchspace_type;
    using distance_type = Dist;
    using label_type = size_t;
    using graph_type = Graph;
    using data_type = Data;

  private:
    distance_type distance_;
    external_id_type counter_{0};
    std::unique_ptr<ParentIndex> index_{nullptr};
    std::unordered_map<label_type, std::vector<external_id_type>> label_to_external_;
    std::unordered_map<external_id_type, label_type> external_to_label_;

    template <class Labels>
    void
    prepare_added_id_by_label(const Labels& labels, std::vector<external_id_type>& adds) {
        for (const auto l : labels) {
            if (label_to_external_.find(l) == label_to_external_.end()) {
                label_to_external_.insert({l, std::vector<external_id_type>{}});
            }

            size_t new_external_id = counter_++;
            label_to_external_[l].push_back(new_external_id);
            external_to_label_.insert({new_external_id, l});
            adds.push_back(new_external_id);
        }
    }

  public:
    template <typename Labels, typename ThreadPoolProto>
    MultiMutableVamanaIndex(
        const VamanaBuildParameters& parameters,
        Data data,
        const Labels& labels,
        Dist distance_function,
        ThreadPoolProto threadpool_proto,
        svs::logging::logger_ptr logger = svs::logging::get()
    )
        : distance_(std::move(distance_function)) {
        std::vector<external_id_type> adds;
        adds.reserve(labels.size());
        prepare_added_id_by_label(labels, adds);
        index_ = std::make_unique<ParentIndex>(
            parameters,
            std::move(data),
            std::move(adds),
            distance_,
            std::move(threadpool_proto),
            std::move(logger)
        );
    }

    template <typename Labels, typename ThreadPoolProto>
    MultiMutableVamanaIndex(
        Graph graph,
        Data data,
        Idx entry_point,
        Dist distance_function,
        const Labels& labels,
        ThreadPoolProto threadpool_proto,
        svs::logging::logger_ptr logger = svs::logging::get()
    )
        : distance_(std::move(distance_function)) {
        std::vector<external_id_type> adds;
        adds.reserve(labels.size());
        prepare_added_id_by_label(labels, adds);

        index_ = std::make_unique<ParentIndex>(
            std::move(graph),
            std::move(data),
            entry_point,
            distance_,
            std::move(adds),
            std::move(threadpool_proto),
            std::move(logger)
        );
    }

    /// @brief Constructor for post re-load dynamic vamana index.
    /// This constructor takes external IDs in translator as labels.
    /// The span of internal ID's in translator should be exactly ``[0, data.size())`.
    template <threads::ThreadPool Pool>
    MultiMutableVamanaIndex(
        const VamanaIndexParameters& config,
        data_type data,
        graph_type graph,
        const Dist& distance_function,
        IDTranslator translator,
        Pool threadpool,
        svs::logging::logger_ptr logger = svs::logging::get()
    )
        : distance_(std::move(distance_function)) {
        // Create labels where labels = translator.external_ids
        std::vector<label_type> labels(translator.size());
        std::transform(
            translator.begin(),
            translator.end(),
            labels.begin(),
            [](const auto& ext_int) { return ext_int.first; }
        );

        std::vector<external_id_type> adds;
        adds.reserve(translator.size());
        prepare_added_id_by_label(labels, adds);

        // create a remapped translator where external_id == internal_id
        IDTranslator remapped_translator;
        remapped_translator.insert(adds, threads::UnitRange<Idx>(0, adds.size()));

        index_ = std::make_unique<ParentIndex>(
            config,
            std::move(data),
            std::move(graph),
            distance_,
            remapped_translator,
            std::move(threadpool),
            std::move(logger)
        );
    }

    const auto& get_label_to_external_lookup() const { return label_to_external_; }
    const auto& get_external_to_label_lookup() const { return external_to_label_; }
    const ParentIndex& get_parent_index() const { return *index_; }

    template <typename Query>
    double get_distance(label_type label, const Query& query) const {
        double best = INVALID_DISTANCE;
        auto it = label_to_external_.find(label);

        if (it != label_to_external_.end()) {
            auto& vectors = (*it).second;
            for (auto each : vectors) {
                best = std::min(
                    best,
                    index_->get_distance(each, query),
                    [](const double a, const double b) {
                        if (std::isnan(a))
                            return false;
                        if (std::isnan(b))
                            return true;
                        return compare{}(a, b);
                    }
                );
            }
        }

        return best;
    }

    template <data::ImmutableMemoryDataset Points, typename Labels>
    std::vector<external_id_type>
    add_points(const Points& points, const Labels& labels, bool reuse_empty = false) {
        const size_t num_points = points.size();
        const size_t num_labels = labels.size();
        if (num_points != num_labels) {
            throw ANNEXCEPTION(
                "Number of points ({}) not equal to the number of external ids ({})!",
                num_points,
                num_labels
            );
        }

        std::vector<external_id_type> adds;
        adds.reserve(num_labels);
        prepare_added_id_by_label(labels, adds);
        index_->add_points(points, adds, reuse_empty);

        return adds;
    }

    template <typename T> void delete_entries(const T& labels) {
        std::vector<external_id_type> deletes;

        for (auto& label : labels) {
            auto it = label_to_external_.find(label);
            if (it != label_to_external_.end()) {
                auto& externals = (*it).second;
                deletes.insert(deletes.end(), externals.begin(), externals.end());
                for (auto& ext : externals) {
                    external_to_label_.erase(ext);
                }
                label_to_external_.erase(it);
            }
        }
        index_->delete_entries(deletes);
    }

    template <typename Query>
    void search(
        const Query& query,
        scratchspace_type& scratch,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) const {
        index_->search(query, scratch, cancel);
    }

    template <typename I, data::ImmutableMemoryDataset Queries>
    void search(
        QueryResultView<I> results,
        const Queries& queries,
        const search_parameters_type& sp,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) {
        auto& borrow_threadpool = index_->get_threadpool_handle();

        threads::parallel_for(
            borrow_threadpool,
            threads::StaticPartition{queries.size()},
            [&](const auto is, uint64_t SVS_UNUSED(tid)) {
                size_t num_neighbors = results.n_neighbors();
                size_t batch_size =
                    std::max(num_neighbors, sp.buffer_config_.get_search_window_size());

                // use batch iterator to search
                for (auto i : is) {
                    auto batch_iterator = make_batch_iterator(queries.get_datum(i), 10);
                    batch_iterator.next(batch_size, cancel);
                    size_t j{0};
                    for (auto& res : batch_iterator) {
                        if (j == num_neighbors) {
                            break;
                        }
                        results.set(res, i, j++);
                    }

                    for (; j < num_neighbors; ++j) {
                        // insert default neighbor if not enough
                        results.set(Neighbor<label_type>{}, i, j);
                    }
                }
            }
        );

        return;
    }

    void compact(Idx batch_size = 1'000) { index_->compact(batch_size); }

    void consolidate() { index_->consolidate(); }

    template <typename QueryType>
    auto make_batch_iterator(
        std::span<const QueryType> query,
        size_t extra_search_buffer_capacity = svs::UNSIGNED_INTEGER_PLACEHOLDER
    ) const {
        return MultiBatchIterator(*this, query, extra_search_buffer_capacity);
    }

    void set_threadpool(threads::ThreadPoolHandle threadpool) {
        index_->set_threadpool(std::move(threadpool));
    }

    ///
    /// @brief Destroy the original thread pool and set to the provided one.
    ///
    /// @param threadpool An acceptable thread pool.
    ///
    /// @copydoc threadpool_requirements
    ///
    template <threads::ThreadPool Pool>
    void set_threadpool(Pool threadpool)
        requires(!std::is_same_v<Pool, threads::ThreadPoolHandle>)
    {
        set_threadpool(threads::ThreadPoolHandle(std::move(threadpool)));
    }

    ///
    /// @brief Return the current thread pool handle.
    ///
    threads::ThreadPoolHandle& get_threadpool_handle() {
        return index_->get_threadpool_handle();
    }

    bool has_id(size_t e) const {
        return label_to_external_.find(e) != label_to_external_.end();
    }

    size_t size() const { return index_->size(); }

    size_t labelcount() const { return label_to_external_.size(); }

    // scrathspace from parent index
    scratchspace_type scratchspace(const search_parameters_type& sp) const {
        return index_->scratchspace(sp);
    }

    // scrathspace from parent index
    scratchspace_type scratchspace() const { return scratchspace(get_search_parameters()); }

    // translate internal id -> external id -> label
    label_type translate_internal_id(Idx i) const {
        return external_to_label_.at(index_->translate_internal_id(i));
    }

    const Data& view_data() const { return index_->view_data(); }
    const Graph& view_graph() const { return index_->view_graph(); }

    void reset_performance_parameters() { index_->reset_performance_parameters(); }

    size_t dimensions() const { return index_->dimensions(); }

    void set_search_parameters(const VamanaSearchParameters& parameters) {
        index_->set_search_parameters(parameters);
    }
    VamanaSearchParameters get_search_parameters() const {
        return index_->get_search_parameters();
    }

    void set_construction_window_size(size_t window_size) {
        index_->set_construction_window_size(window_size);
    }
    size_t get_construction_window_size() const {
        return index_->get_construction_window_size();
    }

    void set_max_candidates(size_t max_candidate_pool_size) {
        index_->set_max_candidates(max_candidate_pool_size);
    }
    size_t get_max_candidates() const { return index_->get_max_candidates(); }

    void set_prune_to(size_t prune_to) { index_->set_prune_to(prune_to); }
    size_t get_prune_to() const { return index_->get_prune_to(); }

    void set_alpha(float alpha) { index_->set_alpha(alpha); }
    float get_alpha() const { return index_->get_alpha(); }

    void set_full_search_history(bool use_full_search_history) {
        index_->set_full_search_history(use_full_search_history);
    }
    bool get_full_search_history() const { return index_->get_full_search_history(); }
};

///// Deduction Guides.
// Guide for building.
template <typename Data, typename Dist, typename ExternalIds>
MultiMutableVamanaIndex(
    const VamanaBuildParameters&, Data, const ExternalIds&, Dist, size_t
) -> MultiMutableVamanaIndex<graphs::SimpleBlockedGraph<uint32_t>, Data, Dist>;

template <typename Data, typename Dist, typename ExternalIds, threads::ThreadPool Pool>
MultiMutableVamanaIndex(const VamanaBuildParameters&, Data, const ExternalIds&, Dist, Pool)
    -> MultiMutableVamanaIndex<graphs::SimpleBlockedGraph<uint32_t>, Data, Dist>;

// Guide with logging
template <typename Data, typename Dist, typename ExternalIds, threads::ThreadPool Pool>
MultiMutableVamanaIndex(
    const VamanaBuildParameters&,
    Data,
    const ExternalIds&,
    Dist,
    Pool,
    svs::logging::logger_ptr
) -> MultiMutableVamanaIndex<graphs::SimpleBlockedGraph<uint32_t>, Data, Dist>;

template <
    typename GraphLoader,
    typename DataLoader,
    typename Distance,
    typename ThreadPoolProto>
auto auto_multi_dynamic_assemble(
    const std::filesystem::path& config_path,
    GraphLoader&& graph_loader,
    DataLoader&& data_loader,
    Distance distance,
    ThreadPoolProto threadpool_proto,
    // Set this to `true` to use the identity map for ID translation.
    // This allows us to read files generated by the static index construction routines
    // to easily benchmark the static versus dynamic implementation.
    //
    // This is an internal API and should not be considered officially supported nor stable.
    bool debug_load_from_static = false,
    svs::logging::logger_ptr logger = svs::logging::get()
) {
    // Load the dataset
    auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
    auto data = svs::detail::dispatch_load(SVS_FWD(data_loader), threadpool);

    // Load the graph.
    auto graph = svs::detail::dispatch_load(SVS_FWD(graph_loader), threadpool);

    // Make sure the data and the graph have the same size.
    auto datasize = data.size();
    auto graphsize = graph.n_nodes();
    if (datasize != graphsize) {
        throw ANNEXCEPTION(
            "Reloaded data has {} nodes while the graph has {} nodes!", datasize, graphsize
        );
    }
    auto [parameters, translator] = lib::load_from_disk<detail::VamanaStateLoader>(
        config_path, debug_load_from_static, datasize
    );

    // Make sure that the translator covers all the IDs in the graph and data.
    auto translator_size = translator.size();
    if (translator_size != datasize) {
        throw ANNEXCEPTION(
            "Translator has {} IDs but should have {}", translator_size, datasize
        );
    }

    for (size_t i = 0; i < datasize; ++i) {
        if (!translator.has_internal(i)) {
            throw ANNEXCEPTION("Translator is missing internal id {}", i);
        }
    }

    return MultiMutableVamanaIndex{
        parameters,
        std::move(data),
        std::move(graph),
        std::move(distance),
        std::move(translator),
        std::move(threadpool),
        std::move(logger)};
}

} // namespace svs::index::vamana
