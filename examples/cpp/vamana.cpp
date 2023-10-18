//! [Example All]

//! [Includes]
// SVS Dependencies
#include "svs/orchestrators/vamana.h"  // bulk of the dependencies required.
#include "svs/core/recall.h"           // Convenient k-recall@n computation.
#include "svs/extensions/vamana/lvq.h" // LVQ extensions for the Vamana index.

// Alternative main definition
#include "svsmain.h"

// stl
#include <map>
#include <string>
#include <string_view>
#include <vector>
//! [Includes]

//! [Helper Utilities]
double run_recall(
    svs::Vamana& index,
    const svs::data::SimpleData<float>& queries,
    const svs::data::SimpleData<uint32_t>& groundtruth,
    size_t search_window_size,
    size_t num_neighbors,
    std::string_view message = ""
) {
    index.set_search_window_size(search_window_size);
    auto results = index.search(queries, num_neighbors);
    double recall = svs::k_recall_at_n(groundtruth, results, num_neighbors, num_neighbors);
    if (!message.empty()) {
        fmt::print("[{}] ", message);
    }
    fmt::print("Windowsize = {}, Recall = {}\n", search_window_size, recall);
    return recall;
}

const bool DEBUG = false;
void check(double expected, double got, double eps = 0.001) {
    double diff = std::abs(expected - got);
    if constexpr (DEBUG) {
        fmt::print("Expected {}. Got {}\n", expected, got);
    } else {
        if (diff > eps) {
            throw ANNEXCEPTION("Expected ", expected, ". Got ", got, '!');
        }
    }
}
//! [Helper Utilities]

// Alternative main definition
int svs_main(std::vector<std::string> args) {
    //! [Argument Extraction]
    const size_t nargs = args.size();
    if (nargs != 4) {
        throw ANNEXCEPTION("Expected 3 arguments. Instead, got ", nargs, '!');
    }
    const std::string& data_vecs = args.at(1);
    const std::string& query_vecs = args.at(2);
    const std::string& groundtruth_vecs = args.at(3);
    //! [Argument Extraction]

    //! [Build Parameters]
    auto parameters = svs::index::vamana::VamanaBuildParameters{
        1.2,  // alpha
        64,   // graph max degree
        128,  // search window size
        1024, // max candidate pool size
        60,   // prune to degree
        true, // full search history
    };
    //! [Build Parameters]

    //! [Index Build]
    size_t num_threads = 4;
    svs::Vamana index = svs::Vamana::build<float>(
        parameters, svs::VectorDataLoader<float>(data_vecs), svs::DistanceL2(), num_threads
    );
    //! [Index Build]

    //! [Load Aux]
    // Load the queries and ground truth.
    auto queries = svs::load_data<float>(query_vecs);
    auto groundtruth = svs::load_data<uint32_t>(groundtruth_vecs);
    //! [Load Aux]

    //! [Perform Queries]
    index.set_search_window_size(30);
    svs::QueryResult<size_t> results = index.search(queries, 10);
    double recall = svs::k_recall_at_n(groundtruth, results);
    check(0.8064, recall);
    //! [Perform Queries]

    //! [Search Window Size]
    auto expected_recall =
        std::map<size_t, double>({{10, 0.5337}, {20, 0.7129}, {30, 0.8064}, {40, 0.8651}});
    for (auto windowsize : {10, 20, 30, 40}) {
        recall = run_recall(index, queries, groundtruth, windowsize, 10, "Sweep");
        check(expected_recall.at(windowsize), recall);
    }
    //! [Search Window Size]

    //! [Saving]
    index.save("example_config", "example_graph", "example_data");
    //! [Saving]

    //! [Loading]
    // We can reload an index from a previously saved set of files.
    index = svs::Vamana::assemble<float>(
        "example_config",
        svs::GraphLoader("example_graph"),
        svs::VectorDataLoader<float>("example_data"),
        svs::DistanceType::L2,
        4 // num_threads
    );

    recall = run_recall(index, queries, groundtruth, 30, 10, "Reload");
    check(0.8065, recall);
    //! [Loading]

    //! [Compressed Loader]
    // Quantization
    size_t padding = 32;
    namespace lvq = svs::quantization::lvq;

    // Wrap the compressor object in a lazy functor.
    // This will defer loading and compression of the LVQ dataset until the threadpool
    // used in the index has been created.
    auto compressor = svs::lib::Lazy([=](svs::threads::ThreadPool auto& threadpool) {
        auto data = svs::VectorDataLoader<float, 128>("example_data").load();
        return lvq::LVQDataset<8, 0, 128>::compress(data, threadpool, padding);
    });
    //! [Compressed Loader]

    //! [Search Compressed]
    index = svs::Vamana::assemble<float>(
        "example_config", svs::GraphLoader("example_graph"), compressor, svs::DistanceL2()
    );

    recall = run_recall(index, queries, groundtruth, 30, 10, "Compressed Load");
    check(0.8065, recall);
    //! [Search Compressed]

    //! [Build Index Compressed]
    // Compressed building
    index =
        svs::Vamana::build<float>(parameters, compressor, svs::DistanceL2(), num_threads);
    recall = run_recall(index, queries, groundtruth, 30, 10, "Compressed Build");
    check(0.8076, recall);
    //! [Build Index Compressed]

    //! [Only Loading]
    // We can reload an index from a previously saved set of files.
    index = svs::Vamana::assemble<float>(
        "example_config",
        svs::GraphLoader("example_graph"),
        svs::VectorDataLoader<float>("example_data"),
        svs::DistanceType::L2,
        4 // num_threads
    );
    //! [Only Loading]

    //! [Set n-threads]
    index.set_num_threads(4);
    //! [Set n-threads]

    return 0;
}

// Special main providing some helpful utilties.
SVS_DEFINE_MAIN();
//! [Example All]
