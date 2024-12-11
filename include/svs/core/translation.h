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

// Index translation.

// svs
#include "svs/lib/algorithms.h"
#include "svs/lib/boundscheck.h"
#include "svs/lib/exception.h"
#include "svs/lib/file.h"
#include "svs/lib/misc.h"
#include "svs/lib/narrow.h"
#include "svs/lib/readwrite.h"
#include "svs/lib/saveload.h"
#include "svs/lib/threads.h"

#include "svs/third-party/fmt.h"

// tsl
#include "tsl/robin_map.h"

// stl
#include <functional>
#include <iterator>
#include <string_view>

namespace svs {

class IDTranslator {
  public:
    using internal_id_type = uint32_t;
    using external_id_type = uint64_t;

    using const_iterator =
        tsl::robin_map<external_id_type, internal_id_type>::const_iterator;
    using value_type = typename const_iterator::value_type;

    // Construct the identity transformation of size `n`.
    struct Identity {
        Identity() = delete;
        Identity(size_t n)
            : n_{n} {}
        size_t n_;
    };

    IDTranslator() = default;
    IDTranslator(Identity tag) {
        auto ids = threads::UnitRange<size_t>{0, tag.n_};
        insert(ids, ids);
    }

    ///
    /// @brief Return the number of translations.
    ///
    size_t size() const {
        if constexpr (checkbounds_v) {
            const size_t e2i = external_to_internal_.size();
            const size_t i2e = internal_to_external_.size();

            if (e2i != i2e) {
                throw ANNEXCEPTION("Size mismatch! E2I is {} while I2E is {}!", e2i, i2e);
            }
        }
        return external_to_internal_.size();
    }

    ///
    /// @brief Insert the two ranges.
    ///
    /// @param external Container implementing forward iteration of the external IDs to add.
    /// @param internal Container implementing forward iteration of the internal IDs to add.
    /// @param check Check that none of the external and internal ids have an assigned
    ///     mapping yet **and** both only contain unique elements. Only safe to set to
    ///     ``false`` if this holds true.
    ///
    /// If any of the checks associated with the ``check`` parameter fail, the container
    /// is left unmodified.
    ///
    template <class External, class Internal>
    void insert(const External& external, const Internal& internal, bool check = true) {
        insert(external.begin(), external.end(), internal.begin(), internal.end(), check);
    }

    template <class ExtBegin, class ExtEnd, class IntBegin, class IntEnd>
    void insert(
        const ExtBegin& ext_begin,
        const ExtEnd& ext_end,
        const IntBegin& int_begin,
        const IntEnd& int_end,
        bool check = true
    ) {
        // Make sure the two iterators are the same length.
        auto external_count = std::distance(ext_begin, ext_end);
        auto internal_count = std::distance(int_begin, int_end);
        if (external_count != internal_count) {
            throw ANNEXCEPTION(
                "Length of external IDs is {} while the length of internal IDs is {}!",
                external_count,
                internal_count
            );
        }

        // Perform checks before actually modifying data structures to prevent a partially
        // completed operation.
        if (check) {
            if (!lib::all_unique(ext_begin, ext_end)) {
                throw ANNEXCEPTION("External IDs contain repeat elements!");
            }
            if (!lib::all_unique(int_begin, int_end)) {
                throw ANNEXCEPTION("Internal IDs contain repeat elements!");
            }
            check_external_free(ext_begin, ext_end);
            check_internal_free(int_begin, int_end);
        }

        // Now, we actually begin the insertion.
        auto i = int_begin;
        auto e = ext_begin;
        while (i != int_end || e != ext_end) {
            insert_translation(*e, *i);
            ++e;
            ++i;
        }
    }

    template <typename Ext, typename Int>
    void insert_translation(Ext external_id, Int internal_id) {
        external_to_internal_[external_id] = lib::narrow<internal_id_type>(internal_id);
        internal_to_external_[internal_id] = lib::narrow<external_id_type>(external_id);
    }

    ///
    /// @brief Return whether the external ID exists.
    ///
    /// @param e The external ID to check.
    ///
    bool has_external(external_id_type e) const {
        return external_to_internal_.contains(e);
    }

    ///
    /// @brief Return whether the internal ID exists.
    ///
    /// @param e The internal ID to check.
    ///
    bool has_internal(internal_id_type e) const {
        return internal_to_external_.contains(e);
    }

    ///
    /// @brief Return the internal ID mapped to by the external ID.
    ///
    /// @param e The external ID to translate to an internal ID.
    ///
    internal_id_type get_internal(external_id_type e) const {
        return external_to_internal_.at(e);
    }

    ///
    /// @brief Return the external ID mapped to by the internal ID.
    ///
    /// @param i The internal ID to translate to an external ID.
    ///
    external_id_type get_external(internal_id_type i) const {
        return internal_to_external_.at(i);
    }

    ///
    /// @brief Return a start forward iterator over the external->internal IDs.
    ///
    const_iterator begin() const { return external_to_internal_.begin(); }

    ///
    /// @brief Return an end forward iterator over the external->internal IDs.
    ///
    const_iterator end() const { return external_to_internal_.end(); }

    ///
    /// @brief Remap the internal ID.
    ///
    /// Assumptions:
    /// * `from` exists.
    /// * `to` does not exist.
    ///
    void remap_internal_id(internal_id_type from, internal_id_type to) {
        assert(has_internal(from));
        assert(!has_internal(to));

        auto itr = internal_to_external_.find(from);
        auto external = itr->second;

        // Updating the internal-to-external ID is easy.
        external_to_internal_[external] = to;
        internal_to_external_.erase(itr);
        internal_to_external_.insert({to, external});
    }

    ///
    /// @brief Delete entries from internal IDs.
    ///
    /// @param internal_ids A container with the internal ids to delete. Must implement a
    ///     forward iterator. Furthermore, all entries must be unique (not checked).
    /// @param check Check if all ids actually exist in the translation table. This is only
    ///     safe to set to ``false`` if it can be guarenteed that all entries in
    ///     ``internal_ids`` exist in the table.
    ///
    /// Note, if ``check == true`` and an internal id is found not to exist, the underlying
    /// translation tables will not be modified.
    ///
    template <typename Internal>
    void delete_internal(const Internal& internal_ids, bool check = true) {
        // First pass - ensure that all IDs to remove actually exist.
        if (check) {
            check_internal_exist(internal_ids.begin(), internal_ids.end());
        }

        for (const auto i : internal_ids) {
            auto e = get_external(i);
            internal_to_external_.erase(i);
            external_to_internal_.erase(e);
        }
    }

    ///
    /// @brief Delete entries from external IDs.
    ///
    /// @param external_ids A container with the external ids to delete. Must implement a
    ///     forward iterator. Furthermore, all entries must be unique (not checked).
    /// @param check Check if all ids actually exist in the translation table. This is only
    ///     safe to set to ``false`` if it can be guarenteed that all entries in
    ///     ``external_ids`` exist in the table.
    ///
    /// Note, if ``check == true`` and an external id is found not to exist, the underlying
    /// translation tables will not be modified.
    ///
    template <typename External>
    void delete_external(const External& external_ids, bool check = true) {
        // First pass - ensure that all IDs to remove actually exist.
        if (check) {
            check_external_exist(external_ids.begin(), external_ids.end());
        }

        for (const auto e : external_ids) {
            auto i = get_internal(e);
            internal_to_external_.erase(i);
            external_to_internal_.erase(e);
        }
    }

    ///
    /// @brief Ensure that **none** of the external ids in the iterator exist yet.
    ///
    /// @param begin Start forward iterator to the external IDs to check.
    /// @param end End forward iterator to the external IDs to check.
    ///
    template <class Begin, class End>
    void check_external_free(const Begin& begin, const End& end) const {
        check(begin, end, external_to_internal_, "Index already contains external");
    }

    ///
    /// @brief Ensure that **all** of the external ids in the iterator exist.
    ///
    /// @param begin Start forward iterator to the external IDs to check.
    /// @param end End forward iterator to the external IDs to check.
    ///
    template <class Begin, class End>
    void check_external_exist(const Begin& begin, const End& end) const {
        check(
            begin,
            end,
            external_to_internal_,
            "Index does not contain external",
            std::logical_not()
        );
    }

    ///
    /// @brief Ensure that **none** of the internal ids in the iterator exist yet.
    ///
    /// @param begin Start forward iterator to the internal IDs to check.
    /// @param end End forward iterator to the internal IDs to check.
    ///
    template <class Begin, class End>
    void check_internal_free(const Begin& begin, const End& end) const {
        check(begin, end, internal_to_external_, "Index already contains internal");
    }

    ///
    /// @brief Ensure that **all** of the internal ids in the iterator exist.
    ///
    /// @param begin Start forward iterator to the internal IDs to check.
    /// @param end End forward iterator to the internal IDs to check.
    ///
    template <class Begin, class End>
    void check_internal_exist(const Begin& begin, const End& end) const {
        check(
            begin,
            end,
            internal_to_external_,
            "Index does not contain internal",
            std::logical_not()
        );
    }

    ///// Saving and Loading
    static constexpr std::string_view kind = "external to internal id translation";
    static constexpr std::string_view serialization_schema =
        "external_to_internal_translation";
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);

    lib::SaveTable save(const lib::SaveContext& ctx) const {
        auto filename = ctx.generate_name("id_translation", "binary");
        // Save the translations to a file.
        auto stream = lib::open_write(filename);
        for (auto i = begin(), iend = end(); i != iend; ++i) {
            // N.B.: Apparently `std::pair` of integers is not trivially copyable ...
            lib::write_binary(stream, i->first);
            lib::write_binary(stream, i->second);
        }
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {{"kind", kind},
             {"num_points", lib::save(size())},
             {"external_id_type", lib::save(datatype_v<external_id_type>)},
             {"internal_id_type", lib::save(datatype_v<internal_id_type>)},
             {"filename", lib::save(filename.filename())}}
        );
    }

    static IDTranslator load(const lib::LoadTable& table) {
        if (kind != lib::load_at<std::string>(table, "kind")) {
            throw ANNEXCEPTION("Mismatched kind!");
        }

        constexpr std::string_view external_id_name = name<datatype_v<external_id_type>>();
        constexpr std::string_view internal_id_name = name<datatype_v<internal_id_type>>();
        if (external_id_name != lib::load_at<std::string>(table, "external_id_type")) {
            throw ANNEXCEPTION("Mismatched external id types!");
        }
        if (internal_id_name != lib::load_at<std::string>(table, "internal_id_type")) {
            throw ANNEXCEPTION("Mismatched internal id types!");
        }

        // Now that we've more-or-less validated the metadata, time to start loading
        // the points.
        auto num_points = lib::load_at<size_t>(table, "num_points");
        auto translator = IDTranslator{};
        auto resolved = table.resolve_at("filename");
        auto stream = lib::open_read(resolved);
        for (size_t i = 0; i < num_points; ++i) {
            auto external_id = lib::read_binary<external_id_type>(stream);
            auto internal_id = lib::read_binary<internal_id_type>(stream);
            translator.insert_translation(external_id, internal_id);
        }
        return translator;
    }

  private:
    template <class Begin, class End, class Map, class Modifier = lib::identity>
    void check(
        const Begin& begin,
        const End& end,
        const Map& map,
        const char* message,
        Modifier modify = lib::identity()
    ) const {
        for (auto i = begin; i != end; ++i) {
            if (modify(map.contains(*i))) {
                throw ANNEXCEPTION("{} ID {}!", message, *i);
            }
        }
    }

    tsl::robin_map<external_id_type, internal_id_type> external_to_internal_{};
    tsl::robin_map<internal_id_type, external_id_type> internal_to_external_{};
};

} // namespace svs
