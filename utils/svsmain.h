/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
 */
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
