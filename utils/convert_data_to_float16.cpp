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
#include <fstream>
#include <iostream>
#include <string>

#include "svs/core/io.h"
#include "svs/lib/float16.h"
#include "svsmain.h"

int svs_main(std::vector<std::string> args) {
    if (args.size() != 4) {
        std::cout << "Specify the right parameters: input index, output index, "
                     "vector_type: 0 for SVS data, 1 for fvecs, 2 for fbin"
                  << std::endl;
        return 1;
    }
    const std::string& filename_f32 = args[1];
    const std::string& filename_f16 = args[2];
    const size_t file_type = std::stoull(args[3]);

    if (file_type == 0) {
        std::cout << "Converting SVS data!" << std::endl;
        auto reader = svs::io::v1::NativeFile{filename_f32}.reader(svs::lib::Type<float>());
        auto writer = svs::io::NativeFile{filename_f16}.writer(
            svs::lib::Type<svs::Float16>(), reader.ndims()
        );

        for (auto i : reader) {
            writer << i;
        }
    } else if (file_type == 1) {
        std::cout << "Converting Vecs data!" << std::endl;
        auto reader = svs::io::vecs::VecsReader<float>{filename_f32};
        auto writer = svs::io::vecs::VecsWriter<svs::Float16>{filename_f16, reader.ndims()};
        for (auto i : reader) {
            writer << i;
        }
    } else if (file_type == 2) {
        std::cout << "Converting Bin data!" << std::endl;
        auto reader = svs::io::binary::BinaryReader<float>{filename_f32};
        auto writer = svs::io::binary::BinaryWriter<svs::Float16>{
            filename_f16, reader.nvectors(), reader.ndims()};
        for (auto i : reader) {
            writer << i;
        }
    }

    return 0;
}

SVS_DEFINE_MAIN();
