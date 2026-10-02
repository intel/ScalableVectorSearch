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

#include "graph_metrics_config.h"

#include "svs/concurrent/dynamic_index.h"
#include "svs/core/data/view.h"
#include "svs/index/vamana/dynamic_index.h"
#include "svs/index/vamana/extensions.h"
#include "svs/lib/float16.h"
#include "svs/lib/threads.h"

#include "fmt/core.h"
#include "fmt/ranges.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

namespace cc = svs::index::vamana::concurrent;
using Idx = uint32_t;
// Reverse-neighbor lists inherit the graph allocator. One mmap per small list
// can exhaust vm.max_map_count when cleanup splits merged mappings. Keep the
// concurrent, grow-stable blocked layout, but allocate through the heap.
using Graph = cc::graphs::SimpleGraph<Idx, cc::SegmentedBlocked<std::allocator<Idx>>>;
using Data = cc::SegmentedBlockedData<float>;
using NonConcurrentGraph =
    svs::graphs::SimpleGraph<Idx, svs::data::Blocked<std::allocator<Idx>>>;
using NonConcurrentData = svs::data::BlockedData<float>;
template <typename Distance, bool Concurrent>
using Index = std::conditional_t<
    Concurrent,
    cc::MutableVamanaIndex<Graph, Data, Distance>,
    svs::index::vamana::
        MutableVamanaIndex<NonConcurrentGraph, NonConcurrentData, Distance>>;

using graph_metrics::Input;
using graph_metrics::Json;
using graph_metrics::Options;

struct BuildPlan {
    bool concurrent;
    std::string_view method;
    size_t batch_size;
    size_t insertion_calls;
    size_t caller_threads;
    size_t worker_threads;
};

std::string_view index_type(bool concurrent) {
    return concurrent ? "svs::index::vamana::concurrent::MutableVamanaIndex"
                      : "svs::index::vamana::MutableVamanaIndex";
}

std::vector<BuildPlan> build_plans(const Options& options) {
    std::vector<BuildPlan> plans;
    for (bool concurrent : {true, false}) {
        const auto type = concurrent ? "concurrent" : "non_concurrent";
        if (options.graph_type != "both" && options.graph_type != type) {
            continue;
        }
        for (std::string_view method : options.build_methods) {
            if (!concurrent && method == "sample_by_sample") {
                continue;
            }
            const size_t batch_size = method == "single_batch"       ? options.total
                                      : method == "sample_by_sample" ? 1
                                                                     : options.batch_size;
            const size_t calls =
                options.total / batch_size + (options.total % batch_size != 0);
            // Only concurrent singleton calls overlap. Batches are submitted in
            // source order, using the index's worker pool inside each call.
            const size_t callers = method == "sample_by_sample" ? options.threads : 1;
            const size_t workers = method == "sample_by_sample" ? 1 : options.threads;
            plans.push_back({concurrent, method, batch_size, calls, callers, workers});
        }
    }
    return plans;
}

constexpr std::string_view help = R"(Usage: graph_metrics --config FILE.json
       graph_metrics --self-test

All dataset definitions and run parameters come from FILE.json. See
graph_metrics_config.json for an example configuration.
Relative source, synthetic-cache, and output paths resolve beside the config file.
threads may be a positive integer or "auto" (CPUs available to this process).

File inputs use little-endian vecs layout with uint32 dimensions before every row.
Coordinates may be uint8, int8, float16, or float32 and are indexed as float32.
Synthetic inputs support normal or uniform distributions, are generated once into
float32 vecs files, and are validated against saved metadata and SHA-256 on reuse.
Mismatched, incomplete, or corrupted synthetic files are regenerated automatically.
Generation, checksum validation, and loading are excluded from index build time.

Every build starts empty. sample_by_sample uses parallel singleton callers and is
concurrent-only. single_batch inserts all rows in one call; batches inserts every
row in sequential calls of batch_size rows, with parallel work inside each call.
Optional entry-point recomputation reuses each singleton/batched graph unchanged.

The report groups datasets -> indexes -> builds, with field descriptions first.
Actual forward edges determine reciprocity and directed shortest paths. Reverse
edge supersets are never used. Every asymmetry maximum reports its source vector
index. Shortest-path means include reachable entries at zero and exclude
unreachable vertices. Progress goes to stderr or the configured output_log.
)";

template <typename T> float source_float(T value) {
    if constexpr (std::is_same_v<T, svs::Float16>) {
        const auto bits = std::bit_cast<uint16_t>(value);
        const auto exponent = bits & 0x7c00;
        if (exponent == 0x7c00) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (exponent == 0) {
            // Preserve IEEE half subnormals, which Float16's fast scalar
            // conversion otherwise flushes to zero.
            const float magnitude = std::ldexp(static_cast<float>(bits & 0x03ff), -24);
            return (bits & 0x8000) ? -magnitude : magnitude;
        }
    }
    return static_cast<float>(value);
}

template <typename T>
void load_vecs(const Input& input, svs::data::SimpleData<float>& points) {
    std::ifstream stream{input.path, std::ios::binary};
    std::vector<T> row(points.dimensions());
    graph_metrics::Sha256 hash;
    for (size_t i = 0; i < points.size(); ++i) {
        uint32_t dimensions = 0;
        if (!stream.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions)) ||
            dimensions != points.dimensions() ||
            !stream.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(T))) {
            throw std::runtime_error(
                fmt::format("Invalid or truncated vector {} in {}", i, input.path.string())
            );
        }
        hash.update(&dimensions, sizeof(dimensions));
        hash.update(row.data(), row.size() * sizeof(T));
        auto destination = points.get_datum(i);
        for (size_t d = 0; d < row.size(); ++d) {
            const float value = source_float(row[d]);
            if (!std::isfinite(value)) {
                throw std::runtime_error(fmt::format(
                    "Non-finite coordinate at vector {}, dimension {} in {}",
                    i,
                    d,
                    input.path.string()
                ));
            }
            destination[d] = value;
        }
    }
    graph_metrics::require(
        hash.finish() == input.sha256,
        "Dataset changed after checksum validation: " + input.path.string()
    );
}

svs::data::SimpleData<float> load_points(const Options& options, const Input& input) {
    fmt::print(
        stderr,
        "Loading {}: {} vectors, dimensions={}...\n",
        input.name,
        options.total,
        options.dimensions
    );
    auto points = svs::data::SimpleData<float>(options.total, options.dimensions);
    if (input.source_type == "uint8") {
        load_vecs<uint8_t>(input, points);
    } else if (input.source_type == "int8") {
        load_vecs<int8_t>(input, points);
    } else if (input.source_type == "float16") {
        load_vecs<svs::Float16>(input, points);
    } else {
        load_vecs<float>(input, points);
    }
    return points;
}

// A sorted CSR copy of ACTUAL forward edges. Sorting only this copy preserves the
// constructed graph and allows reciprocal-edge tests in O(log(out-degree)).
// Call only once writers have joined; a per-node seqlock is not a global snapshot.
struct ForwardGraph {
    std::vector<size_t> offsets;
    std::vector<Idx> edges;

    size_t size() const { return offsets.size() - 1; }

    std::span<const Idx> neighbors(size_t vertex) const {
        return std::span<const Idx>{edges}.subspan(
            offsets[vertex], offsets[vertex + 1] - offsets[vertex]
        );
    }

    template <typename GraphType>
    explicit ForwardGraph(const GraphType& graph)
        : offsets(graph.n_nodes() + 1, 0) {
        for (size_t u = 0; u < graph.n_nodes(); ++u) {
            offsets[u + 1] = offsets[u] + graph.get_node_degree(static_cast<Idx>(u));
        }
        edges.resize(offsets.back());
        for (size_t u = 0; u < graph.n_nodes(); ++u) {
            const auto adjacent = graph.get_node(static_cast<Idx>(u));
            std::copy(adjacent.begin(), adjacent.end(), edges.begin() + offsets[u]);
            auto first = edges.begin() + offsets[u];
            auto last = edges.begin() + offsets[u + 1];
            std::sort(first, last);
            if (std::adjacent_find(first, last) != last) {
                throw std::runtime_error(fmt::format("Duplicate edge at vertex {}", u));
            }
            for (Idx v : neighbors(u)) {
                if (v >= size() || v == u) {
                    throw std::runtime_error(fmt::format("Invalid edge {} -> {}", u, v));
                }
            }
        }
    }
};

double fraction(uint64_t numerator, uint64_t denominator) {
    return denominator == 0
               ? 0.0
               : static_cast<double>(numerator) / static_cast<double>(denominator);
}

struct Metrics {
    size_t vertices = 0;
    size_t edges = 0;
    size_t reachable = 0;
    size_t unreciprocated = 0;
    size_t max_outgoing_count = 0;
    double max_outgoing_fraction = 0;
    size_t max_incoming_count = 0;
    double max_incoming_fraction = 0;
    std::optional<Idx> max_outgoing_count_vertex;
    std::optional<Idx> max_outgoing_fraction_vertex;
    std::optional<Idx> max_incoming_count_vertex;
    std::optional<Idx> max_incoming_fraction_vertex;
    double average_hops = 0;
    size_t maximum_hops = 0;
    double average_out_degree = 0;
    size_t maximum_out_degree = 0;
    std::vector<size_t> out_degree_histogram;
};

template <typename T>
void update_maximum(T value, Idx vertex, T& maximum, std::optional<Idx>& maximum_vertex) {
    // Vertices are visited in ascending order: retain the first vertex on ties,
    // including an all-zero maximum. An empty graph leaves the vertex null.
    if (!maximum_vertex || value > maximum) {
        maximum = value;
        maximum_vertex = vertex;
    }
}

Metrics measure_edges(const ForwardGraph& graph) {
    Metrics result;
    result.vertices = graph.size();
    result.edges = graph.edges.size();
    std::vector<size_t> in_degree(graph.size(), 0);
    std::vector<size_t> incoming_asymmetry(graph.size(), 0);
    for (size_t u = 0; u < graph.size(); ++u) {
        size_t outgoing_asymmetry = 0;
        const auto adjacent = graph.neighbors(u);
        for (Idx v : adjacent) {
            ++in_degree[v];
            const auto reverse = graph.neighbors(v);
            if (!std::binary_search(reverse.begin(), reverse.end(), static_cast<Idx>(u))) {
                ++outgoing_asymmetry;
                ++incoming_asymmetry[v];
            }
        }
        result.unreciprocated += outgoing_asymmetry;
        update_maximum(
            outgoing_asymmetry,
            static_cast<Idx>(u),
            result.max_outgoing_count,
            result.max_outgoing_count_vertex
        );
        update_maximum(
            fraction(outgoing_asymmetry, adjacent.size()),
            static_cast<Idx>(u),
            result.max_outgoing_fraction,
            result.max_outgoing_fraction_vertex
        );
    }
    for (size_t v = 0; v < graph.size(); ++v) {
        update_maximum(
            incoming_asymmetry[v],
            static_cast<Idx>(v),
            result.max_incoming_count,
            result.max_incoming_count_vertex
        );
        update_maximum(
            fraction(incoming_asymmetry[v], in_degree[v]),
            static_cast<Idx>(v),
            result.max_incoming_fraction,
            result.max_incoming_fraction_vertex
        );
    }
    return result;
}

// Reuse edge/asymmetry measurements when only the entry point changes.
void measure_reachability(
    const ForwardGraph& graph, std::span<const Idx> entry_points, Metrics& result
) {
    result.maximum_hops = 0;
    result.maximum_out_degree = 0;
    result.out_degree_histogram.clear();
    constexpr Idx unseen = std::numeric_limits<Idx>::max();
    std::vector<Idx> hops(graph.size(), unseen);
    std::vector<Idx> queue;
    queue.reserve(graph.size());
    for (Idx entry : entry_points) {
        if (entry >= graph.size()) {
            throw std::runtime_error("Entry point is outside the graph");
        }
        if (hops[entry] == unseen) {
            hops[entry] = 0;
            queue.push_back(entry);
        }
    }
    uint64_t total_hops = 0;
    uint64_t total_degree = 0;
    for (size_t head = 0; head < queue.size(); ++head) {
        const Idx u = queue[head];
        total_hops += hops[u];
        result.maximum_hops = std::max(result.maximum_hops, size_t{hops[u]});
        const auto adjacent = graph.neighbors(u);
        const size_t degree = adjacent.size();
        total_degree += degree;
        result.maximum_out_degree = std::max(result.maximum_out_degree, degree);
        if (result.out_degree_histogram.size() <= degree) {
            result.out_degree_histogram.resize(degree + 1, 0);
        }
        ++result.out_degree_histogram[degree];
        for (Idx v : adjacent) {
            if (hops[v] == unseen) {
                hops[v] = hops[u] + 1;
                queue.push_back(v);
            }
        }
    }
    result.reachable = queue.size();
    result.average_hops = fraction(total_hops, result.reachable);
    result.average_out_degree = fraction(total_degree, result.reachable);
}

Metrics measure(const ForwardGraph& graph, std::span<const Idx> entry_points) {
    auto result = measure_edges(graph);
    measure_reachability(graph, entry_points, result);
    return result;
}

template <typename IndexType>
void insert_points(
    IndexType& index,
    const svs::data::SimpleData<float>& points,
    const Options& options,
    const BuildPlan& plan
) {
    fmt::print(
        stderr,
        "Inserting {} vectors, up to {} per call, from {} caller(s), {} worker(s) per "
        "call...\n",
        options.total,
        plan.batch_size,
        plan.caller_threads,
        plan.worker_threads
    );
    const size_t step = std::min(plan.batch_size, options.total);
    std::atomic<size_t> next{0};
    std::atomic<size_t> completed{0};
    std::atomic<size_t> calls{0};
    std::atomic<bool> failed{false};
    std::mutex error_mutex;
    std::exception_ptr error;
    std::vector<std::thread> workers;
    workers.reserve(plan.caller_threads);
    const auto insert = [&] {
        try {
            while (!failed.load(std::memory_order_relaxed)) {
                const size_t first = next.fetch_add(step, std::memory_order_relaxed);
                if (first >= options.total) {
                    break;
                }
                const size_t last = first + std::min(step, options.total - first);
                const auto ids = svs::threads::UnitRange<size_t>{first, last};
                const auto batch = svs::data::make_const_view(points, ids);
                index.add_points(batch, ids);
                calls.fetch_add(1, std::memory_order_relaxed);
                const size_t count =
                    completed.fetch_add(last - first, std::memory_order_relaxed) + last -
                    first;
                if (count % 10'000 == 0 || count == options.total) {
                    fmt::print(stderr, "Inserted {}/{}\n", count, options.total);
                }
            }
        } catch (...) {
            failed.store(true, std::memory_order_relaxed);
            std::lock_guard lock{error_mutex};
            if (!error) {
                error = std::current_exception();
            }
        }
    };
    try {
        if (plan.caller_threads == 1) {
            // Non-concurrent indexes are never called from overlapping writers.
            insert();
        } else {
            for (size_t i = 0; i < plan.caller_threads; ++i) {
                workers.emplace_back(insert);
            }
        }
    } catch (...) {
        failed.store(true, std::memory_order_relaxed);
        for (auto& worker : workers) {
            worker.join();
        }
        throw;
    }
    for (auto& worker : workers) {
        worker.join();
    }
    if (error) {
        std::rethrow_exception(error);
    }
    if (completed.load() != options.total || calls.load() != plan.insertion_calls) {
        throw std::runtime_error("Insertion workload did not match the requested build plan"
        );
    }
}

template <typename IndexType> struct BuiltIndex {
    std::unique_ptr<IndexType> index;
    double index_build_time_seconds;
};

template <typename Distance, bool Concurrent>
BuiltIndex<Index<Distance, Concurrent>> build_index(
    const Options& options, const Input& input, Distance distance, const BuildPlan& plan
) {
    auto points = load_points(options, input);

    // Time only index construction: input I/O/conversion/generation is finished.
    // Include index storage allocation and copying, construction, all additions,
    // and writer joins. Stop before validation, graph measurement, and cleanup.
    const auto build_start = std::chrono::steady_clock::now();
    using IndexType = Index<Distance, Concurrent>;
    typename IndexType::data_type empty_data(0, options.dimensions);
    const auto ids = svs::threads::UnitRange<size_t>{0, 0};
    const svs::index::vamana::VamanaBuildParameters parameters{
        options.alpha,
        options.degree,
        options.window,
        options.max_candidates,
        options.degree,
        options.use_full_search_history};
    auto logger =
        std::make_shared<spdlog::logger>("graph_metrics", svs::logging::stderr_sink());
    logger->set_level(spdlog::level::warn);
    // Inline internal work avoids the native pool's mutex serializing
    // independent concurrent singleton callers. Batch calls use worker threads.
    auto pool = plan.method == "sample_by_sample"
                    ? svs::threads::ThreadPoolHandle{svs::threads::SequentialThreadPool{}}
                    : svs::threads::ThreadPoolHandle{
                          svs::threads::as_threadpool(plan.worker_threads)};
    std::unique_ptr<IndexType> index;
    fmt::print(stderr, "Starting empty; every vector will be inserted via add_points...\n");
    if constexpr (Concurrent) {
        index = std::make_unique<IndexType>(
            parameters, std::move(empty_data), ids, distance, std::move(pool), logger
        );
    } else {
        // The non-concurrent build-from-data constructor requires a medoid.
        // Its graph/data constructor accepts empty storage without building.
        // Routing slot zero is only a placeholder: the first add_points call
        // fills it (and every other row in that batch) before graph traversal.
        index = std::make_unique<IndexType>(
            NonConcurrentGraph{0, options.degree},
            std::move(empty_data),
            Idx{0},
            distance,
            ids,
            std::move(pool),
            logger
        );
        index->set_alpha(options.alpha);
        index->set_construction_window_size(options.window);
        index->set_max_candidates(options.max_candidates);
        index->set_prune_to(options.degree);
        index->set_full_search_history(options.use_full_search_history);
    }
    if (index->size() != 0 || index->view_graph().n_nodes() != 0) {
        throw std::runtime_error("Every insertion method must start with zero vertices");
    }
    insert_points(*index, points, options, plan);
    const double build_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start)
            .count();
    // This workload performs no deletions or failed insertions. Every allocated
    // graph vertex must therefore be live; internal IDs need not equal external IDs.
    if (index->size() != options.total || index->view_graph().n_nodes() != options.total) {
        throw std::runtime_error("Final index size does not equal configured vectors");
    }
    return {std::move(index), build_seconds};
}

std::string json_string(std::string_view text) {
    std::string result = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') {
            result += '\\';
            result += static_cast<char>(c);
        } else if (c < 0x20) {
            result += fmt::format("\\u{:04x}", c);
        } else {
            result += static_cast<char>(c);
        }
    }
    result += '"';
    return result;
}

// JSON has no comments: emit this dictionary as the first member of the document.
void print_field_descriptions() {
    constexpr std::pair<std::string_view, std::string_view> descriptions[] = {
        {"field_descriptions",
         "Field definitions for datasets, their indexes, and each index's builds."},
        {"configuration_file",
         "Absolute path of the JSON configuration used for this run."},
        {"datasets",
         "One object per selected dataset; metadata is shared by all its index types and "
         "builds."},
        {"dataset_name", "Unique dataset name from the configuration."},
        {"indexes",
         "Index implementation groups within a dataset, each containing its build "
         "results."},
        {"index_type",
         "Fully qualified C++ class name, without template arguments: "
         "svs::index::vamana::concurrent::MutableVamanaIndex or "
         "svs::index::vamana::MutableVamanaIndex."},
        {"builds",
         "Build-method results within one index implementation; recomputed-entry variants "
         "reuse their base graph."},
        {"build_method",
         "sample_by_sample: concurrent only, start empty and add every vector with "
         "singleton calls; "
         "single_batch: start empty and insert all rows in one add_points call; "
         "batches: start empty and insert all rows in fixed-size calls, with a smaller "
         "final batch "
         "if needed. sample_by_sample_recompute_entry_point and "
         "batches_recompute_entry_point reuse the respective base graph with an entry "
         "computed over all indexed vectors by the builder's compute_entry_point routine "
         "(nearest vector to the mean, using squared L2 for both L2 and MIP indexes)."},
        {"reused_from_build_method",
         "Base build_method whose graph is reused, or null for a fresh build. Only the "
         "entry used to measure reachability and shortest paths changes."},
        {"index_build_time_seconds",
         "Monotonic wall-clock seconds for index storage allocation/copying, construction, "
         "all add_points calls, and writer joins. Recomputed-entry variants report the "
         "base build time plus entry_point_recompute_time_seconds, measured separately "
         "without rebuilding the graph. Excludes source "
         "loading/conversion/generation, "
         "graph metrics, JSON output, and index cleanup."},
        {"entry_point_recompute_time_seconds",
         "Wall-clock seconds for compute_entry_point on all indexed vectors, including "
         "its worker-pool startup/join; zero for base builds. Already included in "
         "index_build_time_seconds."},
        {"dataset_parameters",
         "Input provenance and selection; values describe the data actually loaded."},
        {"dataset_parameters.path", "Absolute path of the file containing the vectors."},
        {"dataset_parameters.source_kind", "file or synthetic, as configured."},
        {"dataset_parameters.format",
         "vecs: little-endian uint32 dimension per row followed by typed coordinates."},
        {"dataset_parameters.source_type",
         "Coordinate type in the source file; float16 occupies two bytes even in .fvecs "
         "files."},
        {"dataset_parameters.index_data_type",
         "Coordinate type used by the index after conversion: float32."},
        {"dataset_parameters.source_vectors",
         "Number of rows in the complete source file."},
        {"dataset_parameters.selected_vectors",
         "Number of source vectors used by every build for this dataset."},
        {"dataset_parameters.dimensions", "Coordinates per vector."},
        {"dataset_parameters.distance",
         "L2 is squared Euclidean distance; MIP maximizes inner product."},
        {"dataset_parameters.selection",
         "prefix: first selected_vectors rows in file order, including saved synthetic "
         "data."},
        {"dataset_parameters.normalized",
         "False: no normalization is applied to the loaded vectors."},
        {"dataset_parameters.seed",
         "MT19937 seed for synthetic vectors; null for file datasets."},
        {"dataset_parameters.distribution",
         "normal or uniform for synthetic vectors; null for file datasets."},
        {"dataset_parameters.generation",
         "Synthetic metadata (generator version, dimensions, count, type, distribution, "
         "parameters, seed); null for file datasets. Saved beside synthetic data as "
         "<path>.meta.json, with an additional sha256 field."},
        {"dataset_parameters.sha256",
         "SHA-256 of the selected prefix's raw vecs bytes, including row headers. "
         "Rechecked while loading every build; covers the complete file for synthetic "
         "data."},
        {"configuration", "Effective construction and insertion parameters for this run."},
        {"configuration.insertion_batch_size",
         "Maximum rows per add_points call; total for single_batch."},
        {"configuration.insertion_calls",
         "Number of add_points calls from an empty index: total for sample_by_sample, "
         "one for single_batch, ceil(total / batch_size) for batches."},
        {"configuration.insertion_caller_threads",
         "Caller threads submitting additions: configured threads for concurrent singleton "
         "insertion, one for either batch method."},
        {"configuration.insertion_worker_threads",
         "Workers inside each add_points call: one for singletons, configured threads for "
         "either batch method."},
        {"configuration.entry_point_recompute_threads",
         "Workers used to recompute the entry point; zero for base builds."},
        {"configuration.threads",
         "Configured thread count; both batch methods use this many "
         "workers, "
         "concurrent singletons use this many callers."},
        {"configuration.degree", "Maximum out-degree and prune_to."},
        {"configuration.window", "Construction search window."},
        {"configuration.max_candidates", "Construction candidate limit."},
        {"configuration.alpha", "Configured pruning parameter for the dataset's distance."},
        {"configuration.use_full_search_history",
         "Whether construction uses full search history."},
        {"entry_points_internal",
         "Internal entry IDs used for these metrics after all writers join: original "
         "index entries for base builds, or the recomputed entry for reused-graph "
         "variants."},
        {"vertices", "Number of live vertices, including unreachable vertices."},
        {"edges",
         "Number of actual directed edges, including edges at unreachable vertices."},
        {"reachable_vertices",
         "Vertices reachable from any entry via directed forward edges, including "
         "entries."},
        {"unreachable_vertices", "vertices minus reachable_vertices."},
        {"unreachable_fraction", "unreachable_vertices / vertices."},
        {"unreciprocated_edges",
         "Actual edges u->v without an actual v->u. Reverse-edge superset records are "
         "ignored."},
        {"edge_asymmetry_fraction",
         "unreciprocated_edges / edges; a bidirectional pair counts as two edges."},
        {"max_outgoing_asymmetry_count",
         "Maximum per-vertex number of outgoing edges without a reverse edge, over all "
         "vertices."},
        {"max_outgoing_asymmetry_fraction",
         "Maximum per-vertex outgoing asymmetric count / out-degree, over all vertices."},
        {"max_incoming_asymmetry_count",
         "Maximum per-vertex number of incoming edges without a reverse edge, over all "
         "vertices."},
        {"max_incoming_asymmetry_fraction",
         "Maximum per-vertex incoming asymmetric count / actual in-degree, over all "
         "vertices."},
        {"max_outgoing_asymmetry_count_vertex",
         "Vertex attaining the outgoing count maximum."},
        {"max_outgoing_asymmetry_fraction_vertex",
         "Vertex attaining the outgoing fraction maximum; computed independently of "
         "count."},
        {"max_incoming_asymmetry_count_vertex",
         "Vertex attaining the incoming count maximum."},
        {"max_incoming_asymmetry_fraction_vertex",
         "Vertex attaining the incoming fraction maximum; computed independently of "
         "count."},
        {"*_vertex.internal_id",
         "Internal graph ID. Ties choose the smallest internal ID; the entire vertex "
         "object is null for an empty graph."},
        {"*_vertex.vector_index",
         "Zero-based row in the source dataset, or generated vector number. Translated "
         "from internal ID after concurrent insertion."},
        {"average_shortest_path_hops",
         "Mean directed, unweighted shortest path from the nearest entry; reachable "
         "vertices only, entries at zero."},
        {"maximum_shortest_path_hops",
         "Maximum directed shortest path from the nearest entry over reachable vertices."},
        {"reachable_average_out_degree", "Mean actual out-degree over reachable vertices."},
        {"reachable_maximum_out_degree",
         "Maximum actual out-degree over reachable vertices."},
        {"reachable_out_degree_histogram",
         "Element d counts reachable vertices with out-degree d."},
        {"zero_denominators",
         "All fractions and means with zero denominator are reported as zero."},
    };
    fmt::print("  \"field_descriptions\": {{\n");
    for (size_t i = 0; i < std::size(descriptions); ++i) {
        fmt::print(
            "    {}: {}{}\n",
            json_string(descriptions[i].first),
            json_string(descriptions[i].second),
            i + 1 == std::size(descriptions) ? "" : ","
        );
    }
    fmt::print("  }},\n");
}

template <typename Translate>
std::string vertex_json(std::optional<Idx> vertex, Translate&& translate) {
    if (!vertex) {
        return "null";
    }
    return fmt::format(
        "{{\"internal_id\": {}, \"vector_index\": {}}}", *vertex, translate(*vertex)
    );
}

void print_dataset_header(const Options& options, const Input& input) {
    fmt::print(
        "    {{\n"
        "      \"dataset_name\": {},\n"
        "      \"dataset_parameters\": {{\n"
        "        \"path\": {},\n"
        "        \"source_kind\": {},\n"
        "        \"format\": \"vecs\",\n"
        "        \"source_type\": {},\n"
        "        \"index_data_type\": \"float32\",\n"
        "        \"source_vectors\": {},\n"
        "        \"selected_vectors\": {},\n"
        "        \"dimensions\": {},\n"
        "        \"distance\": {},\n"
        "        \"selection\": \"prefix\",\n"
        "        \"normalized\": false,\n"
        "        \"seed\": {},\n"
        "        \"distribution\": {},\n"
        "        \"generation\": {},\n"
        "        \"sha256\": {}\n"
        "      }},\n"
        "      \"indexes\": [\n",
        json_string(input.name),
        json_string(input.path.string()),
        json_string(input.kind),
        json_string(input.source_type),
        input.source_vectors,
        options.total,
        options.dimensions,
        json_string(input.distance),
        input.generation.is_null() ? "null" : input.generation.at("seed").dump(),
        input.generation.is_null() ? "null" : input.generation.at("distribution").dump(),
        input.generation.dump(),
        json_string(input.sha256)
    );
}

template <typename Translate>
void print_build_report(
    const Options& options,
    const BuildPlan& plan,
    double index_build_time_seconds,
    std::optional<double> entry_point_recompute_seconds,
    std::span<const Idx> entry_points,
    const Metrics& metrics,
    Translate&& translate
) {
    fmt::print(
        "            {{\n"
        "              \"build_method\": {},\n"
        "              \"reused_from_build_method\": {},\n"
        "              \"index_build_time_seconds\": {},\n"
        "              \"entry_point_recompute_time_seconds\": {},\n"
        "              \"configuration\": {{\n"
        "                \"threads\": {},\n"
        "                \"degree\": {},\n"
        "                \"window\": {},\n"
        "                \"max_candidates\": {},\n"
        "                \"alpha\": {},\n"
        "                \"use_full_search_history\": {},\n"
        "                \"insertion_batch_size\": {},\n"
        "                \"insertion_calls\": {},\n"
        "                \"insertion_caller_threads\": {},\n"
        "                \"insertion_worker_threads\": {},\n"
        "                \"entry_point_recompute_threads\": {}\n"
        "              }},\n"
        "              \"entry_points_internal\": [{}],\n",
        json_string(
            entry_point_recompute_seconds
                ? fmt::format("{}_recompute_entry_point", plan.method)
                : std::string{plan.method}
        ),
        entry_point_recompute_seconds ? json_string(plan.method) : "null",
        index_build_time_seconds + entry_point_recompute_seconds.value_or(0),
        entry_point_recompute_seconds.value_or(0),
        options.threads,
        options.degree,
        options.window,
        options.max_candidates,
        options.alpha,
        options.use_full_search_history,
        plan.batch_size,
        plan.insertion_calls,
        plan.caller_threads,
        plan.worker_threads,
        entry_point_recompute_seconds ? options.threads : 0,
        fmt::join(entry_points, ", ")
    );
    fmt::print(
        "              \"vertices\": {},\n"
        "              \"edges\": {},\n"
        "              \"reachable_vertices\": {},\n"
        "              \"unreachable_vertices\": {},\n"
        "              \"unreachable_fraction\": {},\n"
        "              \"unreciprocated_edges\": {},\n"
        "              \"edge_asymmetry_fraction\": {},\n"
        "              \"max_outgoing_asymmetry_count\": {},\n"
        "              \"max_outgoing_asymmetry_count_vertex\": {},\n"
        "              \"max_outgoing_asymmetry_fraction\": {},\n"
        "              \"max_outgoing_asymmetry_fraction_vertex\": {},\n"
        "              \"max_incoming_asymmetry_count\": {},\n"
        "              \"max_incoming_asymmetry_count_vertex\": {},\n"
        "              \"max_incoming_asymmetry_fraction\": {},\n"
        "              \"max_incoming_asymmetry_fraction_vertex\": {},\n",
        metrics.vertices,
        metrics.edges,
        metrics.reachable,
        metrics.vertices - metrics.reachable,
        fraction(metrics.vertices - metrics.reachable, metrics.vertices),
        metrics.unreciprocated,
        fraction(metrics.unreciprocated, metrics.edges),
        metrics.max_outgoing_count,
        vertex_json(metrics.max_outgoing_count_vertex, translate),
        metrics.max_outgoing_fraction,
        vertex_json(metrics.max_outgoing_fraction_vertex, translate),
        metrics.max_incoming_count,
        vertex_json(metrics.max_incoming_count_vertex, translate),
        metrics.max_incoming_fraction,
        vertex_json(metrics.max_incoming_fraction_vertex, translate)
    );
    fmt::print(
        "              \"average_shortest_path_hops\": {},\n"
        "              \"maximum_shortest_path_hops\": {},\n"
        "              \"reachable_average_out_degree\": {},\n"
        "              \"reachable_maximum_out_degree\": {},\n"
        "              \"reachable_out_degree_histogram\": [{}]\n"
        "            }}",
        metrics.average_hops,
        metrics.maximum_hops,
        metrics.average_out_degree,
        metrics.maximum_out_degree,
        fmt::join(metrics.out_degree_histogram, ", ")
    );
    // Preserve the completed report even if index cleanup subsequently fails.
    if (std::fflush(stdout) != 0) {
        throw std::runtime_error("Failed to flush the JSON report");
    }
}

// Hand-calculated fixtures exercise direction, unreachable vertices, denominators,
// independent count/fraction maxima, multiple sources, and stale reverse records.
void self_test() {
    const auto require = [](bool condition, std::string_view name) {
        if (!condition) {
            throw std::runtime_error(fmt::format("Self-test failed: {}", name));
        }
    };
    Graph graph(8, 3);
    graph.enable_reverse_edges();
    const std::array<std::vector<Idx>, 8> edges{
        {{1, 2, 3}, {0, 2}, {3}, {1, 7}, {5}, {4}, {2}, {2}}};
    for (Idx u = 0; u < edges.size(); ++u) {
        graph.replace_node(u, edges[u]);
    }
    // Pretend every edge also has a reverse edge in the superset. None of these
    // extra records changes the actual forward adjacency used for measurement.
    for (Idx u = 0; u < edges.size(); ++u) {
        for (Idx v : edges[u]) {
            graph.reverse_edges()->record(v, u);
        }
    }
    const ForwardGraph snapshot{graph};
    const auto result = measure(snapshot, std::array<Idx, 1>{0});
    require(result.vertices == 8 && result.edges == 12, "vertex/edge counts");
    require(result.reachable == 5, "directed reachability");
    require(result.unreciprocated == 8, "actual edges, including unreachable sources");
    require(result.max_outgoing_count == 2, "outgoing count maximum");
    require(result.max_outgoing_fraction == 1.0, "outgoing fraction maximum");
    require(result.max_incoming_count == 4, "incoming count maximum");
    require(result.max_incoming_fraction == 1.0, "incoming fraction maximum");
    require(
        result.max_outgoing_count_vertex == 0 && result.max_outgoing_fraction_vertex == 2 &&
            result.max_incoming_count_vertex == 2 &&
            result.max_incoming_fraction_vertex == 2,
        "maximum vertices and smallest-ID tie breaking"
    );
    require(result.average_hops == 1.0 && result.maximum_hops == 2, "BFS distances");
    require(result.average_out_degree == 1.8, "reachable mean out-degree");
    require(result.maximum_out_degree == 3, "reachable maximum out-degree");
    require(
        result.out_degree_histogram == std::vector<size_t>{0, 2, 2, 1},
        "reachable out-degree histogram"
    );

    const auto multiple = measure(snapshot, std::array<Idx, 4>{0, 4, 6, 0});
    require(multiple.reachable == 8, "multiple/duplicate entry points");
    require(
        multiple.average_hops == 0.75 && multiple.maximum_hops == 2,
        "minimum distance from any entry"
    );
    auto changed_entry = result;
    measure_reachability(snapshot, std::array<Idx, 1>{4}, changed_entry);
    require(
        changed_entry.reachable == 2 && changed_entry.average_hops == 0.5 &&
            changed_entry.maximum_hops == 1 && changed_entry.average_out_degree == 1 &&
            changed_entry.maximum_out_degree == 1 &&
            changed_entry.out_degree_histogram == std::vector<size_t>{0, 2},
        "changing entry resets all reachability metrics on the reused graph"
    );
    require(
        changed_entry.edges == result.edges &&
            changed_entry.unreciprocated == result.unreciprocated &&
            changed_entry.max_outgoing_count_vertex == result.max_outgoing_count_vertex &&
            changed_entry.max_incoming_count_vertex == result.max_incoming_count_vertex,
        "changing entry preserves edge/asymmetry measurements"
    );

    // Count maxima occur at 0 (outgoing) and 3 (incoming), but fraction maxima
    // occur at 5 and 4 respectively. Vertex 5 is unreachable from the entry point.
    Graph distinct_maxima(7, 4);
    const std::array<std::vector<Idx>, 7> other_edges{
        {{1, 2, 3, 4}, {0}, {0}, {6}, {}, {3}, {3}}};
    for (Idx u = 0; u < other_edges.size(); ++u) {
        distinct_maxima.replace_node(u, other_edges[u]);
    }
    const auto maxima = measure(ForwardGraph{distinct_maxima}, std::array<Idx, 1>{0});
    require(
        maxima.max_outgoing_count == 2 && maxima.max_outgoing_fraction == 1.0 &&
            maxima.max_incoming_count == 2 && maxima.max_incoming_fraction == 1.0,
        "count and fraction maxima at different vertices"
    );
    require(
        maxima.max_outgoing_count_vertex == 0 && maxima.max_outgoing_fraction_vertex == 5 &&
            maxima.max_incoming_count_vertex == 3 &&
            maxima.max_incoming_fraction_vertex == 4,
        "separate vertices for count and fraction maxima"
    );
    require(
        vertex_json(maxima.max_outgoing_fraction_vertex, [](Idx id) { return 100 + id; }) ==
            "{\"internal_id\": 5, \"vector_index\": 105}",
        "translate maximum vertex to the original dataset row"
    );

    Graph empty_edges(3, 2);
    const auto isolated = measure(ForwardGraph{empty_edges}, std::array<Idx, 1>{0});
    require(isolated.reachable == 1 && isolated.edges == 0, "isolated entry point");
    require(
        isolated.average_hops == 0 && isolated.average_out_degree == 0 &&
            isolated.max_incoming_fraction == 0 && isolated.max_outgoing_fraction == 0 &&
            fraction(0, 0) == 0,
        "zero denominators"
    );
    require(
        isolated.max_outgoing_count_vertex == 0 &&
            isolated.max_incoming_count_vertex == 0 &&
            isolated.max_outgoing_fraction_vertex == 0 &&
            isolated.max_incoming_fraction_vertex == 0,
        "all-zero maxima identify the first vertex"
    );
    const auto no_entry = measure(snapshot, {});
    require(no_entry.reachable == 0 && no_entry.average_hops == 0, "no entry points");
    Graph empty(0, 2);
    const auto empty_result = measure(ForwardGraph{empty}, {});
    require(
        empty_result.vertices == 0 && !empty_result.max_outgoing_count_vertex &&
            !empty_result.max_outgoing_fraction_vertex &&
            !empty_result.max_incoming_count_vertex &&
            !empty_result.max_incoming_fraction_vertex,
        "empty graph has no maximum vertices"
    );
    require(
        vertex_json(
            {}, [](Idx) -> size_t { throw std::runtime_error("Unexpected translation"); }
        ) == "null",
        "empty maximum serializes as null without translation"
    );
    require(
        json_string("path\"\\\n\t") == "\"path\\\"\\\\\\u000a\\u0009\"", "JSON escaping"
    );
    require(
        source_float(int8_t{-128}) == -128.0f && source_float(uint8_t{255}) == 255.0f,
        "signed and unsigned byte conversion"
    );
    const auto half = [](uint16_t bits) { return std::bit_cast<svs::Float16>(bits); };
    require(
        source_float(half(0x3e00)) == 1.5f && source_float(half(0xbe00)) == -1.5f &&
            source_float(half(0x0001)) == 0x1p-24f &&
            source_float(half(0x8001)) == -0x1p-24f &&
            !std::isfinite(source_float(half(0x7c00))) &&
            !std::isfinite(source_float(half(0x7e00))),
        "float16 conversion preserves subnormals and rejects non-finite values"
    );

    // Allocate reverse lists in shuffled order, then free them in vertex order.
    // The previous mmap-backed allocator aborts here under the common Linux
    // vm.max_map_count=65530 limit, even though constructing the graph succeeds.
    {
        constexpr size_t count = 300'000;
        Graph large(count, 1);
        large.enable_reverse_edges();
        std::vector<Idx> order(count);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 generator{42};
        std::shuffle(order.begin(), order.end(), generator);
        for (Idx u : order) {
            large.add_edge(u, (u + 1) % count);
        }
    }
    fmt::print("Graph metric self-tests passed.\n");
}

template <typename Distance, bool Concurrent>
void run_dataset(
    const Options& options, const Input& input, Distance distance, const BuildPlan& plan
) {
    const auto graph_type = index_type(Concurrent);
    fmt::print(stderr, "Run: {} / {} / {}\n", input.name, graph_type, plan.method);
    auto [index, build_seconds] =
        build_index<Distance, Concurrent>(options, input, distance, plan);
    fmt::print(stderr, "Index build time: {:.6f} seconds.\n", build_seconds);
    fmt::print(stderr, "All writers joined. Measuring the forward graph...\n");
    index->experimental_escape_hatch(
        [&](const auto& graph, const auto& data, const auto&, std::span<const Idx> entries
        ) {
            const ForwardGraph snapshot{graph};
            auto metrics = measure(snapshot, entries);
            const auto translate = [&](Idx id) { return index->translate_internal_id(id); };
            print_build_report(
                options, plan, build_seconds, std::nullopt, entries, metrics, translate
            );
            if (options.recompute_entry_point && plan.method != "single_batch") {
                fmt::print(stderr, "Recomputing entry point on the same graph...\n");
                const auto recompute_start = std::chrono::steady_clock::now();
                Idx entry;
                {
                    svs::threads::NativeThreadPool pool{options.threads};
                    entry = svs::lib::narrow<Idx>(
                        svs::index::vamana::extensions::compute_entry_point(data, pool)
                    );
                }
                const double recompute_seconds =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - recompute_start
                    )
                        .count();
                fmt::print(
                    stderr,
                    "Entry point: {} -> {}; recompute time: {:.6f} seconds.\n",
                    fmt::join(entries, ", "),
                    entry,
                    recompute_seconds
                );
                // The utility only measures the built graph. Use the selected entry as
                // the BFS source without mutating the index or its adjacency lists.
                const std::array<Idx, 1> recomputed_entries{entry};
                measure_reachability(snapshot, recomputed_entries, metrics);
                fmt::print(",\n");
                print_build_report(
                    options,
                    plan,
                    build_seconds,
                    recompute_seconds,
                    recomputed_entries,
                    metrics,
                    translate
                );
            }
        }
    );
    fmt::print(stderr, "JSON report flushed. Releasing index...\n");
    index.reset();
    fmt::print(stderr, "Done: {} / {} / {}.\n", input.name, graph_type, plan.method);
}

template <typename Distance>
void dispatch_run(
    const Options& options, const Input& input, Distance distance, const BuildPlan& plan
) {
    if (plan.concurrent) {
        run_dataset<Distance, true>(options, input, distance, plan);
    } else {
        run_dataset<Distance, false>(options, input, distance, plan);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view{argv[1]} == "--help") {
            fmt::print("{}", help);
            return 0;
        }
        if (argc == 2 && std::string_view{argv[1]} == "--self-test") {
            self_test();
            return 0;
        }
        if (argc != 3 || std::string_view{argv[1]} != "--config") {
            throw std::invalid_argument("Usage: graph_metrics --config FILE.json");
        }
        auto config = graph_metrics::read_configuration(argv[2]);
        fmt::print(stderr, "JSON: {}\n", config.output_json.string());
        if (!config.output_log.empty()) {
            fmt::print(stderr, "Log:  {}\n", config.output_log.string());
            graph_metrics::redirect_stream(stderr, config.output_log);
        }
        graph_metrics::prepare_inputs(config);
        // Publish a complete report atomically. Invalid configurations or failed
        // builds leave any previous report intact.
        graph_metrics::TemporaryFile output{config.output_json};
        graph_metrics::redirect_stream(stdout, output.path);
        fmt::print("{{\n");
        print_field_descriptions();
        fmt::print("  \"configuration_file\": {},\n", json_string(config.path.string()));
        fmt::print("  \"datasets\": [\n");
        size_t completed_datasets = 0;
        for (const auto& [effective, input] : config.inputs) {
            const auto plans = build_plans(effective);
            if (completed_datasets++ != 0) {
                fmt::print(",\n");
            }
            print_dataset_header(effective, input);
            size_t completed_indexes = 0;
            for (bool concurrent : {true, false}) {
                if (std::none_of(plans.begin(), plans.end(), [&](const auto& plan) {
                        return plan.concurrent == concurrent;
                    })) {
                    continue;
                }
                if (completed_indexes++ != 0) {
                    fmt::print(",\n");
                }
                fmt::print(
                    "        {{\n          \"index_type\": {},\n          \"builds\": [\n",
                    json_string(index_type(concurrent))
                );
                size_t completed_builds = 0;
                for (const auto& plan : plans) {
                    if (plan.concurrent != concurrent) {
                        continue;
                    }
                    if (completed_builds++ != 0) {
                        fmt::print(",\n");
                    }
                    if (input.distance == "MIP") {
                        dispatch_run(effective, input, svs::DistanceIP{}, plan);
                    } else {
                        dispatch_run(effective, input, svs::DistanceL2{}, plan);
                    }
                }
                fmt::print("\n          ]\n        }}");
            }
            fmt::print("\n      ]\n    }}");
        }
        fmt::print("\n  ]\n}}\n");
        if (std::fflush(stdout) != 0) {
            throw std::runtime_error("Failed to flush the JSON report");
        }
        output.commit(config.output_json);
        fmt::print(stderr, "JSON report: {}\n", config.output_json.string());
        return 0;
    } catch (const std::exception& error) {
        fmt::print(stderr, "Error: {}\n", error.what());
        return 1;
    }
}
