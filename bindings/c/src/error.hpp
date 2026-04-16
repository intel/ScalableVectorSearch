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
#pragma once

#include "svs/c_api/svs_c.h"

#include <stdexcept>
#include <string>

#include <svs/lib/exception.h>

// C API error structure
struct svs_error_desc {
    svs_error_code_t code;
    std::string message;
};

#define SET_ERROR(err, c, msg)      \
    do {                            \
        if (err) {                  \
            (err)->code = (c);      \
            (err)->message = (msg); \
        }                           \
    } while (0)

#define NOT_IMPLEMENTED_IF(cond, msg)                   \
    do {                                                \
        if (cond) {                                     \
            throw svs::c_runtime::not_implemented(msg); \
        }                                               \
    } while (0)

#define INVALID_ARGUMENT_IF(cond, msg)        \
    do {                                      \
        if (cond) {                           \
            throw std::invalid_argument(msg); \
        }                                     \
    } while (0)

#define EXPECT_ARG_IN_RANGE(arg, min_val, max_val)              \
    INVALID_ARGUMENT_IF(                                        \
        (arg) < (min_val) || (arg) > (max_val),                 \
        #arg " should be in range [" #min_val ", " #max_val "]" \
    )

#define EXPECT_ARG_GT_THAN(arg, threshold) \
    INVALID_ARGUMENT_IF((arg) <= (threshold), #arg " should be greater than " #threshold)

#define EXPECT_ARG_GE_THAN(arg, threshold)                                          \
    INVALID_ARGUMENT_IF(                                                            \
        (arg) < (threshold), #arg " should be greater than or equal to " #threshold \
    )

#define EXPECT_ARG_NOT_NULL(arg) \
    INVALID_ARGUMENT_IF((arg) == nullptr, #arg " should not be NULL")

#define EXPECT_ARG_IS_NULL(arg) \
    INVALID_ARGUMENT_IF((arg) != nullptr, #arg " should be NULL")

#define EXPECT_ARG_EQ_TO(actual, expected)                                       \
    INVALID_ARGUMENT_IF(                                                         \
        (actual) != (expected), "Expected " #actual " to be equal to " #expected \
    )

#define EXPECT_ARG_NE_TO(actual, expected)                                           \
    INVALID_ARGUMENT_IF(                                                             \
        (actual) == (expected), "Expected " #actual " to be not equal to " #expected \
    )

namespace svs::c_runtime {

class not_implemented : public std::logic_error {
  public:
    using std::logic_error::logic_error;
};

class invalid_operation : public std::logic_error {
  public:
    using std::logic_error::logic_error;
};

class unsupported_hw : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// A helper to wrap C++ exceptions and convert them to C error codes/messages.
template <typename Callable, typename Result = std::invoke_result_t<Callable>>
Result wrap_exceptions(Callable&& func, svs_error_h err, Result err_res = {}) noexcept {
    try {
        SET_ERROR(err, SVS_OK, "Success");
        return func();
    } catch (const std::invalid_argument& ex) {
        SET_ERROR(err, SVS_ERROR_INVALID_ARGUMENT, ex.what());
        return err_res;
    } catch (const svs::c_runtime::not_implemented& ex) {
        SET_ERROR(err, SVS_ERROR_NOT_IMPLEMENTED, ex.what());
        return err_res;
    } catch (const svs::c_runtime::invalid_operation& ex) {
        SET_ERROR(err, SVS_ERROR_INVALID_OPERATION, ex.what());
        return err_res;
    } catch (const svs::c_runtime::unsupported_hw& ex) {
        SET_ERROR(err, SVS_ERROR_UNSUPPORTED_HW, ex.what());
        return err_res;
    } catch (const svs::lib::ANNException& ex) {
        SET_ERROR(err, SVS_ERROR_GENERIC, ex.what());
        return err_res;
    } catch (const std::runtime_error& ex) {
        SET_ERROR(err, SVS_ERROR_RUNTIME, ex.what());
        return err_res;
    } catch (const std::exception& ex) {
        SET_ERROR(err, SVS_ERROR_UNKNOWN, ex.what());
        return err_res;
    } catch (...) {
        SET_ERROR(err, SVS_ERROR_UNKNOWN, "An unknown error has occurred.");
        return err_res;
    }
}
} // namespace svs::c_runtime
