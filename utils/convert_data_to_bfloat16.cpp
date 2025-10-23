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

#include <fstream>
#include <iostream>
#include <string>

#include "svs/core/io.h"
#include "svs/lib/bfloat16.h"
#include "svsmain.h"

int svs_main(std::vector<std::string> args) {
    if (args.size() != 4) {
        std::cout << "Specify the right parameters: input index, output index, "
                     "vector_type: 0 for SVS data, 1 for fvecs, 2 for fbin"
                  << std::endl;
        return 1;
    }
    const std::string& filename_f32 = args[1];
    const std::string& filename_bf16 = args[2];
    const size_t file_type = std::stoull(args[3]);

    if (file_type == 0) {
        std::cout << "Converting SVS data!" << std::endl;
        auto reader = svs::io::v1::NativeFile{filename_f32}.reader(svs::lib::Type<float>());
        auto writer = svs::io::NativeFile{filename_bf16}.writer(
            svs::lib::Type<svs::BFloat16>(), reader.ndims()
        );

        for (auto i : reader) {
            writer << i;
        }
    } else if (file_type == 1) {
        std::cout << "Converting Vecs data!" << std::endl;
        auto reader = svs::io::vecs::VecsReader<float>{filename_f32};
        auto writer =
            svs::io::vecs::VecsWriter<svs::BFloat16>{filename_bf16, reader.ndims()};
        for (auto i : reader) {
            writer << i;
        }
    } else if (file_type == 2) {
        std::cout << "Converting Bin data!" << std::endl;
        auto reader = svs::io::binary::BinaryReader<float>{filename_f32};
        auto writer = svs::io::binary::BinaryWriter<svs::BFloat16>{
            filename_bf16, reader.nvectors(), reader.ndims()
        };
        for (auto i : reader) {
            writer << i;
        }
    }

    return 0;
}

SVS_DEFINE_MAIN();
