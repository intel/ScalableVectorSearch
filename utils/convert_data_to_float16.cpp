/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
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
                     "vector_type: 0 for SVS data, 1 for fvecs"
                  << std::endl;
        return 1;
    }
    const std::string& filename_f32 = args[1];
    const std::string& filename_f16 = args[2];
    const size_t file_type = std::stoull(args[3]);

    if (file_type == 0) {
        std::cout << "Converting SVS data!" << std::endl;
        auto reader =
            svs::io::v1::NativeFile{filename_f32}.reader(svs::meta::Type<float>());
        auto writer = svs::io::NativeFile{filename_f16}.writer(
            svs::meta::Type<svs::Float16>(), reader.ndims()
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
    }
    return 0;
}

SVS_DEFINE_MAIN();
