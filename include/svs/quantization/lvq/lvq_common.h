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

#pragma once

namespace svs {
namespace quantization {
namespace lvq {

namespace detail {

// Trait to determine if an allocator is blocked or not.
// Used to SFINAE away resizing methods if the allocator is not blocked.
template <typename A> inline constexpr bool is_blocked = false;
template <typename A> inline constexpr bool is_blocked<data::Blocked<A>> = true;

} // namespace detail

enum class LVQStrategyDispatch {
    Auto,       // Choose between sequential and turbo.
    Sequential, // Force Sequential
    Turbo       // Force Turbo
};

///
/// Place-holder to indicate that a given direct compression stores its values as
/// signed integers (taking positive and negative values in accordance with a two-s
/// complement encoding).
///
struct Signed {
    static constexpr std::string_view name = "signed";
};

///
/// Place-holder to indicate that a given direct compression stores its values as
/// unsigned integers.
///
struct Unsigned {
    static constexpr std::string_view name = "unsigned";
};

// Schemas are independent of most type parameters.
// Hoist them as stand-alone variables to they are accessible to the auto load
// matchers as well.
inline constexpr std::string_view one_level_serialization_schema = "one_level_lvq_dataset";
inline constexpr lib::Version one_level_save_version = lib::Version(0, 0, 2);
inline constexpr std::string_view two_level_serialization_schema = "two_level_lvq_dataset";
inline constexpr lib::Version two_level_save_version = lib::Version(0, 0, 3);
inline constexpr std::string_view fallback_serialization_schema = "fallback_dataset";
inline constexpr lib::Version fallback_save_version = lib::Version(0, 0, 0);

enum class DatasetSchema { Compressed, ScaledBiased, Fallback };
///
/// Support for deduction.
///
inline constexpr std::string_view get_schema(DatasetSchema kind) {
    switch (kind) {
        using enum DatasetSchema;
        case Compressed: {
            return "lvq_compressed_dataset";
        }
        case ScaledBiased: {
            return "lvq_with_scaling_constants";
        }
        case Fallback: {
            return "uncompressed_data";
        }
    }
    throw ANNEXCEPTION("Invalid schema!");
}

inline constexpr lib::Version get_current_version(DatasetSchema kind) {
    switch (kind) {
        using enum DatasetSchema;
        case Compressed: {
            return lib::Version(0, 0, 0);
        }
        case ScaledBiased: {
            return lib::Version(0, 0, 3);
        }
        case Fallback: {
            return lib::Version(0, 0, 0);
        }
    }
    throw ANNEXCEPTION("Invalid schema!");
}

struct DatasetSummary {
    static bool check_load_compatibility(std::string_view schema, lib::Version version) {
        using enum DatasetSchema;
        if (schema == get_schema(Compressed) &&
            version == get_current_version(Compressed)) {
            return true;
        }
        if (schema == get_schema(ScaledBiased) &&
            version == get_current_version(ScaledBiased)) {
            return true;
        }
        if (schema == get_schema(Fallback) && version == get_current_version(Fallback)) {
            return true;
        }
        return false;
    }

    static DatasetSummary load(const lib::ContextFreeLoadTable& table) {
        using enum DatasetSchema;
        auto schema = table.schema();
        if (schema == get_schema(Compressed)) {
            return DatasetSummary{
                .kind = Compressed,
                .is_signed =
                    (lib::load_at<std::string>(table, "sign") == lvq::Signed::name),
                .dims = lib::load_at<size_t>(table, "ndims"),
                .bits = lib::load_at<size_t>(table, "bits")};
        }
        if (schema == get_schema(ScaledBiased)) {
            return DatasetSummary{
                .kind = ScaledBiased,
                .is_signed = false, // ScaledBiased always uses unsigned codes.
                .dims = lib::load_at<size_t>(table, "logical_dimensions"),
                .bits = lib::load_at<size_t>(table, "bits")};
        }
        if (schema == get_schema(Fallback)) {
            return DatasetSummary{
                .kind = Fallback,
                .is_signed = false,
                .dims = lib::load_at<size_t>(table, "dims"),
                .bits = 32};
        }
        throw ANNEXCEPTION("Invalid table schema {}!", schema);
    }

    ///// Members
    // The kind of the leaf dataset.
    DatasetSchema kind;
    // Whether each LVQ element is signed.
    bool is_signed;
    // The logical number of dimensions in the dataset.
    size_t dims;
    // The number of bits used for compression.
    size_t bits;
};

} // namespace lvq
} // namespace quantization
} // namespace svs
