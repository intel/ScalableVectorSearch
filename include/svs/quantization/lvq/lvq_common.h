/*
 * Copyright 2023 Intel Corporation
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

#include "svs/quantization/lvq/impl/config.h"
#include "svs/fallback/fallback_mode.h"

namespace svs {
namespace quantization {
namespace lvq {

namespace detail {

// Trait to determine if an allocator is blocked or not.
// Used to SFINAE away resizing methods if the allocator is not blocked.
template <typename A> inline constexpr bool is_blocked = false;
template <typename A> inline constexpr bool is_blocked<data::Blocked<A>> = true;
// template <typename T> inline constexpr bool is_lvq_packing_strategy_v = false;

} // namespace detail// Strategies for storing packed data.
struct Sequential {
    static constexpr std::string_view name() { return "sequential"; }
    static constexpr size_t compute_bytes(size_t nbits, size_t length) {
        return lib::div_round_up(nbits * length, 8);
    }

    // No permutation required.
    static constexpr size_t logical_to_linear(size_t i) { return i; }
    static constexpr size_t linear_to_logical(size_t i) { return i; }
};

// Blockwise strategy.
template <size_t Lanes, size_t ElementsPerLane> struct Turbo {
    static constexpr std::string name() {
        return fmt::format("turbo<{}x{}>", Lanes, ElementsPerLane);
    }
    static constexpr size_t lanes = Lanes;
    static constexpr size_t elements_per_lane = ElementsPerLane;
    static constexpr size_t block_size = Lanes * ElementsPerLane;

    static constexpr size_t compute_bytes(size_t nbits, size_t length) {
        assert(nbits == 4 || nbits == 8);

        size_t block_size_bytes = nbits * block_size / 8;
        size_t num_blocks = lib::div_round_up(length, block_size);
        return block_size_bytes * num_blocks;
    }

    static constexpr size_t logical_to_linear(size_t i) {
        // `a`: Which block we are in.
        // `b`: Tne entry in the block.
        // `c`: The offset in the lane
        // `d`: Which lane.
        auto [a, b] = detail::divrem(i, block_size);
        auto [c, d] = detail::divrem(b, Lanes);
        return block_size * a + ElementsPerLane * d + c;
    }

    static constexpr size_t linear_to_logical(size_t i) {
        // `a`: Which block we are in.
        // `b`: The entry in the block.
        auto [a, b] = detail::divrem(i, block_size);
        auto [c, d] = detail::divrem(b, ElementsPerLane);
        return block_size * a + Lanes * d + c;
    }

    static constexpr size_t num_blocks(size_t count) {
        return lib::round_up_to_multiple_of(count, block_size);
    }
};

namespace detail {

// Trait to identify and dispatch based on the Turbo class itself.
template <typename T> inline constexpr bool is_turbo_like_v = false;
template <typename T> inline constexpr bool is_lvq_packing_strategy_v = false;

template <size_t Lanes, size_t ElementsPerLane>
inline constexpr bool is_turbo_like_v<lvq::Turbo<Lanes, ElementsPerLane>> = true;

template <> inline constexpr bool is_lvq_packing_strategy_v<lvq::Sequential> = true;
template <size_t Lanes, size_t ElementsPerLane>

inline constexpr bool is_lvq_packing_strategy_v<lvq::Turbo<Lanes, ElementsPerLane>> = true;

} // namespace detail

template <typename T>
concept TurboLike = detail::is_turbo_like_v<T>;

template <typename T>
concept UsesSequential = std::is_same_v<typename T::strategy, Sequential>;

template <typename T>
concept UsesTurbo = TurboLike<typename T::strategy>;

enum class LVQStrategyDispatch {
    Auto,       // Choose between sequential and turbo.
    Sequential, // Force Sequential
    Turbo       // Force Turbo
};

template <typename T>
concept LVQPackingStrategy = detail::is_lvq_packing_strategy_v<T>;

// Forward declaration of the primary template
template <
    size_t Primary,
    size_t Residual,
    size_t Extent,
    LVQPackingStrategy Strategy,
    typename Alloc,
    svs::fallback::FallbackBool Fallback>
class LVQDataset;

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

enum class DatasetSchema { Compressed, ScaledBiased };
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
