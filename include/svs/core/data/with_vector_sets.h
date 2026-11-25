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

// svs
#include "svs/concepts/vector_set.h"
#include "svs/lib/exception.h"
#include "svs/lib/saveload.h"

// stl
#include <concepts>
#include <cstddef>
#include <string_view>
#include <utility>

namespace svs::data {

///
/// @brief Pairs a primary dataset (one row per entry, used for graph traversal) with a
/// vector-set dataset (one token-vector set per entry, used for re-ranking).
///
/// Models ``ImmutableMemoryDataset`` by forwarding every member to ``primary()``, so
/// graph search over a ``WithVectorSets`` is identical to graph search over the primary
/// dataset alone. ``get_vector_set(i)`` is the extra accessor a Vamana re-rank extension
/// uses after the graph search has produced a candidate pool.
///
template <typename Primary, ImmutableVectorSetDataset Sets> class WithVectorSets {
  public:
    using value_type = typename Primary::value_type;
    using const_value_type = typename Primary::const_value_type;
    using element_type = typename Primary::element_type;

    // VamanaIndex reads `Data::extent` unconditionally, outside the ImmutableMemoryDataset
    // concept; omitting this breaks every VamanaIndex instantiation over this type.
    static constexpr size_t extent = Primary::extent;

    WithVectorSets(Primary primary, Sets sets)
        : primary_{std::move(primary)}
        , sets_{std::move(sets)} {
        if (primary_.size() != sets_.size()) {
            throw ANNEXCEPTION(
                "Primary dataset has {} entries but vector-set dataset has {} entries",
                primary_.size(),
                sets_.size()
            );
        }
    }

    size_t size() const { return primary_.size(); }
    size_t dimensions() const { return primary_.dimensions(); }
    const_value_type get_datum(size_t i) const { return primary_.get_datum(i); }
    void prefetch(size_t i) const { primary_.prefetch(i); }

    const Primary& primary() const { return primary_; }
    const Sets& vector_sets() const { return sets_; }

    /// The candidate's token-vector set, for re-ranking after graph search.
    typename Sets::const_value_type get_vector_set(size_t i) const {
        return sets_.get_datum(i);
    }

    ///// IO
    //
    // Available only when both `Primary` and `Sets` are themselves saveable/loadable;
    // a non-owning `Sets` (e.g. `ConstVectorSetView`) lacks `save`/`load`, and the
    // requires-clauses below make that a silent non-participant rather than a hard
    // error, so `WithVectorSets` over a view still compiles as a dataset member.

    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
    static constexpr std::string_view serialization_schema = "with_vector_sets";

    static bool check_load_compatibility(std::string_view schema, lib::Version version) {
        return schema == serialization_schema && version == save_version;
    }

    /// @brief Save both members, nested as sub-tables under the ``"primary"`` and
    /// ``"vector_sets"`` keys.
    lib::SaveTable save(const lib::SaveContext& ctx) const
        requires requires(const Primary& p, const Sets& s, const lib::SaveContext& c) {
                     p.save(c);
                     s.save(c);
                 }
    {
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {{"primary", lib::save(primary_, ctx)}, {"vector_sets", lib::save(sets_, ctx)}}
        );
    }

    /// @brief Reload both members with their default-constructed allocators.
    static WithVectorSets load(const lib::LoadTable& table)
        requires requires(const lib::LoadTable& t) {
                     { Primary::load(t) } -> std::same_as<Primary>;
                     { Sets::load(t) } -> std::same_as<Sets>;
                 }
    {
        return WithVectorSets(
            lib::load_at<Primary>(table, "primary"),
            lib::load_at<Sets>(table, "vector_sets")
        );
    }

  private:
    Primary primary_;
    Sets sets_;
};

} // namespace svs::data
