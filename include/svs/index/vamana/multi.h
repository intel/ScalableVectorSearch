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
    /// Random-access iterator to `value_type` over the current batch of results.
    using iterator = typename result_buffer_type::iterator;
    /// Random-access iterator to `const value_type` over the current batch of results.
    using const_iterator = typename result_buffer_type::const_iterator;

    using ParentIndex = typename Index::ParentIndex;

  public:
    MultiBatchIterator(
        Index& index,
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
        results_.clear();
        get_results_from_extra(batch_size);

        while (results_.size() < batch_size && !batch_iterator_.done()) {
            batch_iterator_.next(batch_size, cancel);
            for (auto& result : batch_iterator_) {
                auto label = external_to_label.at(result.id());

                if (returned_.find(label) == returned_.end()) {
                    returned_.insert(label);
                    if (results_.size() < batch_size) {
                        results_.push_back(Neighbor<label_type>{label, result.distance()});
                    } else {
                        extra_results_.push_back(Neighbor<label_type>{
                            label, result.distance()});
                    }
                }
            }
        }
        return;
    }

    void update(std::span<const QueryType> newquery) {
        returned_.clear();
        results_.clear();
        extra_results_.clear();
        batch_iterator_.update(newquery);
    }

    /// @brief Returns an iterator to the beginning of the results.
    iterator begin() { return results_.begin(); }
    /// @brief Returns an iterator to the end of the results.
    iterator end() { return results_.end(); }
    /// @copydoc begin()
    const_iterator begin() const { return results_.begin(); }
    /// @copydoc end()
    const_iterator end() const { return results_.end(); }
    /// @copydoc begin()
    const_iterator cbegin() const { return results_.cbegin(); }
    /// @copydoc begin()
    const_iterator cend() const { return results_.cend(); }

    bool done() const { return batch_iterator_.done() && extra_results_.empty(); }

    std::span<const value_type> contents() const { return lib::as_const_span(results_); }

  private:
    void get_results_from_extra(size_t batch_size) {
        std::sort(
            extra_results_.begin(),
            extra_results_.end(),
            std::greater<Neighbor<label_type>>{}
        );

        while (results_.size() < batch_size && !extra_results_.empty()) {
            auto top = extra_results_.front();
            extra_results_.pop_back();

            if (returned_.find(top.id()) == returned_.end()) {
                returned_.insert(top.id());
                results_.push_back(std::move(top));
            }
        }

        return;
    }

    Index& index_;
    std::unordered_set<label_type> returned_;
    std::vector<Neighbor<label_type>> results_;
    std::vector<Neighbor<label_type>> extra_results_;
    BatchIterator<ParentIndex, QueryType> batch_iterator_;
};

template <typename Data, typename Dist> class MultiMutableVamanaIndex {
  public:
    using Graph = graphs::SimpleBlockedGraph<uint32_t>;
    using ParentIndex = MutableVamanaIndex<Graph, Data, Dist>;
    using Idx = typename ParentIndex::Idx;
    using search_parameters_type = typename ParentIndex::search_parameters_type;
    using external_id_type = typename ParentIndex::external_id_type;
    using distance_type = Dist;
    using label_type = size_t;

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
    template <typename ThreadPoolProto, typename Labels>
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
            logger
        );
    }

    const auto& get_label_to_external_lookup() const { return label_to_external_; }
    const auto& get_external_to_label_lookup() const { return external_to_label_; }
    const ParentIndex& get_parent_index() const { return *index_; }

    template <typename Query>
    double get_distance(label_type label, const Query& query) const {
        double min = std::numeric_limits<double>::max();
        auto it = label_to_external_.find(label);
        if (it != label_to_external_.end()) {
            auto& vectors = (*it).second;
            for (auto each : vectors) {
                min = std::min(min, index_->get_distance(each, query));
            }
        }

        return min;
    }

    template <data::ImmutableMemoryDataset Points, typename Labels>
    std::vector<label_type>
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

    template <typename I, data::ImmutableMemoryDataset Queries>
    void search(
        QueryResultView<I> results,
        const Queries& queries,
        const search_parameters_type& SVS_UNUSED(sp
        ), // In this version, we assume search parameters are the same as index
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) {
        auto& borrow_threadpool = index_->get_threadpool_handle();

        threads::parallel_for(
            borrow_threadpool,
            threads::StaticPartition{queries.size()},
            [&](const auto is, uint64_t SVS_UNUSED(tid)) {
                size_t num_neighbors = results.n_neighbors();

                // use batch iterator to search
                for (auto i : is) {
                    auto batch_iterator = generate_batch_iterator(queries.get_datum(i));
                    batch_iterator.next(num_neighbors, cancel);
                    size_t j{0};
                    for (auto& res : batch_iterator) {
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
    auto generate_batch_iterator(std::span<const QueryType> query) const {
        return MultiBatchIterator(*this, query, 1);
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

    size_t size() const { return label_to_external_.size(); }
};

} // namespace svs::index::vamana
