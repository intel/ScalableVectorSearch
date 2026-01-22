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
#include "svs/c_api/svs_c.h"

#include "error.hpp"

#include <string>

extern "C" svs_error_h svs_error_init() { return new svs_error_desc{SVS_OK, "Success"}; }
extern "C" bool svs_error_ok(svs_error_h err) { return err->code == SVS_OK; }
extern "C" svs_error_code_t svs_error_get_code(svs_error_h err) { return err->code; }
extern "C" const char* svs_error_get_message(svs_error_h err) {
    return err->message.c_str();
}
extern "C" void svs_error_free(svs_error_h err) { delete err; }
