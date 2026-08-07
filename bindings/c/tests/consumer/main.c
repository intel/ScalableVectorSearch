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

/*
 * Smoke test for the installed C API package.
 *
 * This is compiled as C (not C++) on purpose: it proves the shipped headers are
 * valid C and that consuming the exported CMake target does not drag C++ flags
 * or unresolved C++ dependencies into a plain C project. It also reports which
 * compressed storage backends the installed library provides, which is the
 * capability a downstream integration has to branch on today.
 */

#include "svs/c_api/svs_c.h"

#include <stdio.h>
#include <stdlib.h>

/* Report whether a storage backend is available, without treating an absent
 * proprietary backend as a failure. */
static void report(const char* name, svs_storage_h storage, svs_error_h error) {
    if (storage != NULL) {
        printf("%-24s available\n", name);
        svs_storage_free(storage);
        return;
    }

    svs_error_code_t code = svs_error_get_code(error);
    const char* reason = "unavailable";
    if (code == SVS_ERROR_NOT_IMPLEMENTED) {
        reason = "not built in";
    } else if (code == SVS_ERROR_UNSUPPORTED_HW) {
        reason = "unsupported hardware";
    }
    printf("%-24s %s (%s)\n", name, reason, svs_error_get_message(error));
}

int main(void) {
    svs_error_h error = svs_error_create();
    if (error == NULL) {
        fprintf(stderr, "failed to create an error handle\n");
        return EXIT_FAILURE;
    }

    /* Simple storage is part of every build, so treat its absence as fatal:
     * it is the minimum proof that the installed library actually works. */
    svs_storage_h simple = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32, error);
    if (simple == NULL) {
        fprintf(
            stderr, "failed to create simple storage: %s\n", svs_error_get_message(error)
        );
        svs_error_free(error);
        return EXIT_FAILURE;
    }
    printf("%-24s available\n", "simple/float32");
    svs_storage_free(simple);

    report("sq/int8", svs_storage_create_sq(SVS_DATA_TYPE_INT8, error), error);

    report(
        "lvq/int8",
        svs_storage_create_lvq(SVS_DATA_TYPE_INT8, SVS_DATA_TYPE_VOID, error),
        error
    );

    report(
        "leanvec/int8",
        svs_storage_create_leanvec(64, SVS_DATA_TYPE_INT8, SVS_DATA_TYPE_INT8, error),
        error
    );

    svs_error_free(error);
    return EXIT_SUCCESS;
}
