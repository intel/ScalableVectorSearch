#pragma once

#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "svs/lib/exception.h"

namespace svsmain {
namespace detail {
std::vector<std::string> process_args(int argc, char* argv[]) {
    std::vector<std::string> args{};
    for (auto& i : std::span(argv, argc)) {
        args.emplace_back(i);
    }
    return args;
}
} // namespace detail
} // namespace svsmain

#define SVS_DEFINE_MAIN()                                                                \
    int main(int argc, char* argv[]) {                                                   \
        int return_code = 0;                                                             \
        try {                                                                            \
            return_code = svs_main(svsmain::detail::process_args(argc, argv));           \
        } catch (const svs::ANNException& err) {                                         \
            std::cerr << "Application terminated with ANNException: " << err.what()      \
                      << std::endl;                                                      \
            return EXIT_FAILURE;                                                         \
        } catch (const std::exception& err) {                                            \
            std::cerr << "Application terminated with unknown exception: " << err.what() \
                      << std::endl;                                                      \
            return EXIT_FAILURE;                                                         \
        }                                                                                \
        return return_code;                                                              \
    }
