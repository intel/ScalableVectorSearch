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
#include "svs/lib/segmented_vector.h"
#include "svs/lib/threads.h"

#include "svs/third-party/fmt.h"

// tsl
#include "tsl/robin_map.h"

// stl
#include <algorithm>
#include <atomic>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace svs {

namespace detail {

// Storage and slot access for the internal->external table of ``IDTranslatorBase``.
// ``load`` yields ``default_val`` for an index beyond the table, and ``store`` fills the
// slots it has to add below the stored index with its last argument.
template <typename T, bool Concurrent> struct VectorMapTraits;

template <typename T> struct VectorMapTraits<T, false> {
    using value_type = T;
    using vector_type = std::vector<value_type>;

    static value_type load(vector_type const& table, size_t i, value_type default_val) {
        return i < table.size() ? table[i] : default_val;
    }

    static void store(vector_type& table, size_t i, value_type external, value_type def) {
        if (i >= table.size()) {
            table.resize(i + 1, def);
        }
        table[i] = external;
    }
};

// Readers may run without a lock against one writer: ``SegmentedVector`` never relocates a
// slot when it grows, and every slot access after construction is atomic. A plain read or
// write of a slot here would race with those readers.
template <typename T> struct VectorMapTraits<T, true> {
    using value_type = T;
    using vector_type = lib::SegmentedVector<value_type>;

    static value_type load(vector_type const& table, size_t i, value_type default_val) {
        if (i >= table.size()) {
            return default_val;
        }
        // C++20 has no ``std::atomic_ref<const T>``; the cast only removes the constness
        // of the table reference, the slot itself is never a const object.
        return std::atomic_ref<value_type>(const_cast<value_type&>(table[i]))
            .load(std::memory_order_acquire);
    }

    static void
    store(vector_type& table, size_t i, value_type external, value_type default_val) {
        if (i >= table.size()) {
            table.resize(i + 1, default_val);
        }
        std::atomic_ref<value_type>(table[i]).store(external, std::memory_order_release);
    }
};

///
/// @brief Two-way translation between external IDs and internal IDs.
///
/// External IDs map to internal IDs through a hash map. Internal IDs map back through a
/// table indexed by internal ID, so internal IDs are expected to be dense: the table
/// holds one ``external_id_type`` per slot up to the largest internal ID ever stored.
///
/// @tparam Concurrent Selects the storage of the internal-to-external table. Use the
///     aliases ``svs::IDTranslator`` (``false``) and ``svs::ConcurrentIDTranslator``
///     (``true``); the template itself is not part of the public API.
///
/// With ``Concurrent == false`` the translator offers no thread safety beyond concurrent
/// ``const`` access.
///
/// With ``Concurrent == true``:
/// * Mutating methods must be serialized by the caller (one writer at a time).
/// * ``has_internal``, ``get_external`` and ``get_external_or`` may be called without a
///   lock while a writer is running. They observe each slot either before or after the
///   writer's update of that slot.
/// * Every other method reads the external-to-internal hash map and must be synchronized
///   with writers by the caller, e.g. with a shared lock.
///
template <bool Concurrent> class IDTranslatorBase {
  public:
    using internal_id_type = uint32_t;
    using external_id_type = uint64_t;

    using const_iterator =
        tsl::robin_map<external_id_type, internal_id_type>::const_iterator;
    using value_type = typename const_iterator::value_type;

    /// Marks an unmapped slot in the internal-to-external table. Reserved: it is rejected
    /// as an external ID.
    static constexpr external_id_type invalid_external_id =
        std::numeric_limits<external_id_type>::max();

    // Construct the identity transformation of size `n`.
    struct Identity {
        Identity() = delete;
        Identity(size_t n)
            : n_{n} {}
        size_t n_;
    };

    IDTranslatorBase() = default;
    IDTranslatorBase(Identity tag) {
        auto ids = threads::UnitRange<size_t>{0, tag.n_};
        insert(ids, ids);
    }

    ///
    /// @brief Construct from the other variant, keeping every translation.
    ///
    /// @param other The translator to take the translations from.
    ///
    explicit IDTranslatorBase(IDTranslatorBase<!Concurrent> other)
        : external_to_internal_{std::move(other.external_to_internal_)} {
        for (const auto& [external, internal] : external_to_internal_) {
            store_external(internal, external);
        }
    }

    ///
    /// @brief Return the number of translations.
    ///
    size_t size() const { return external_to_internal_.size(); }

    ///
    /// @brief Return the number of bytes allocated for both translation tables.
    ///
    size_t get_memory_usage() const {
        // The external->internal map is approximated by its entries, ignoring load-factor
        // slack. An exact count depends on tsl::robin_map internals:
        //   using bucket_type = tsl::detail_robin_hash::
        //       bucket_entry<std::pair<external_id_type, internal_id_type>, false>;
        //   external_to_internal_.bucket_count() * sizeof(bucket_type)
        return size() * (sizeof(external_id_type) + sizeof(internal_id_type)) +
               internal_to_external_.capacity() * sizeof(external_id_type);
    }

    ///
    /// @brief Estimate ``get_memory_usage()`` for ``n`` translations over slots ``[0, n)``.
    ///
    static constexpr size_t estimate_memory_usage(size_t n) {
        return n * (sizeof(external_id_type) + sizeof(internal_id_type)) +
               n * sizeof(external_id_type);
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
            check_external_not_reserved(ext_begin, ext_end);
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

    ///
    /// @brief Insert one translation without checking for an existing mapping.
    ///
    /// @param external_id The external ID. Must not be ``invalid_external_id``.
    /// @param internal_id The internal ID.
    ///
    /// Throws without modifying the translator if either ID does not fit its ID type or
    /// ``external_id`` is reserved. Only safe if neither ID has a mapping yet.
    ///
    template <typename Ext, typename Int>
    void insert_translation(Ext external_id, Int internal_id) {
        auto e = lib::narrow<external_id_type>(external_id);
        auto i = lib::narrow<internal_id_type>(internal_id);
        // Storing the reserved ID would leave `e` mapped while slot `i` reads as unmapped.
        check_external_not_reserved(&e, &e + 1);
        external_to_internal_[e] = i;
        store_external(i, e);
    }

    ///
    /// @brief Insert mappings, replacing any existing mapping whose current internal
    ///     ID is considered stale by the caller (e.g. the associated slot has been
    ///     marked for deletion but the translator entry was not yet cleaned up).
    ///
    /// Throws if an external ID already maps to a non-stale internal ID.
    ///  The batch is validated in full before any mutation, so on throw the
    /// translation maps are left unmodified.
    ///
    /// @param external Container of external IDs to insert.
    /// @param internal Container of internal IDs to insert (same length as external).
    /// @param is_stale Callable ``bool(internal_id_type)``. Returns true if the given
    ///     internal ID is stale and its existing mapping may be overwritten.
    ///
    template <class External, class Internal, class IsStale>
    void replace_stale_and_insert(
        const External& external, const Internal& internal, IsStale&& is_stale
    ) {
        auto ext_begin = external.begin();
        auto ext_end = external.end();
        auto int_begin = internal.begin();
        auto int_end = internal.end();

        const auto external_count = std::distance(ext_begin, ext_end);
        const auto internal_count = std::distance(int_begin, int_end);
        if (external_count != internal_count) {
            throw ANNEXCEPTION(
                "Length of external IDs is {} while the length of internal IDs is {}!",
                external_count,
                internal_count
            );
        }
        if (!lib::all_unique(ext_begin, ext_end)) {
            throw ANNEXCEPTION("External IDs contain repeat elements!");
        }
        if (!lib::all_unique(int_begin, int_end)) {
            throw ANNEXCEPTION("Internal IDs contain repeat elements!");
        }

        check_external_not_reserved(ext_begin, ext_end);

        // Validation pass
        for (auto e = ext_begin; e != ext_end; ++e) {
            auto found = external_to_internal_.find(*e);
            if (found != external_to_internal_.end() && !is_stale(found->second)) {
                throw ANNEXCEPTION("Index already contains external ID {}!", *e);
            }
        }

        // Commit pass
        auto e = ext_begin;
        auto i = int_begin;
        for (; e != ext_end; ++e, ++i) {
            auto found = external_to_internal_.find(*e);
            if (found != external_to_internal_.end()) {
                store_external(found->second, invalid_external_id);
            }
            insert_translation(*e, *i);
        }
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
        return load_external(e) != invalid_external_id;
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
        auto e = load_external(i);
        if (e == invalid_external_id) {
            throw std::out_of_range("IDTranslator: internal ID not found");
        }
        return e;
    }

    ///
    /// @brief Return the external ID mapped to by the internal ID, or a default.
    ///
    /// @param i The internal ID to translate to an external ID.
    /// @param default_val The value returned when ``i`` has no mapping.
    ///
    external_id_type
    get_external_or(internal_id_type i, external_id_type default_val) const {
        auto e = load_external(i);
        return e == invalid_external_id ? default_val : e;
    }

    ///
    /// @brief Return the internal ID mapped to by the external ID, or a default.
    ///
    /// @param e The external ID to translate to an internal ID.
    /// @param default_val The value returned when ``e`` has no mapping.
    ///
    internal_id_type
    get_internal_or(external_id_type e, internal_id_type default_val) const {
        auto it = external_to_internal_.find(e);
        if (it == external_to_internal_.end()) {
            return default_val;
        }
        return it->second;
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

        auto external = load_external(from);

        // Updating the internal-to-external ID is easy.
        external_to_internal_[external] = to;
        store_external(from, invalid_external_id);
        store_external(to, external);
    }

    ///
    /// @brief Remap the external ID, keeping the same internal ID.
    ///
    /// @param from The existing external ID to rename.
    /// @param to The external ID to rename it to. Must not be ``invalid_external_id``.
    /// @param check Check that ``from`` exists and ``to`` does not yet exist. Only safe
    ///     to set to ``false`` if this holds true.
    ///
    /// Note, if ``check == true`` and either condition fails, or if ``to`` is reserved,
    /// the underlying translation tables will not be modified.
    ///
    void remap_external_id(external_id_type from, external_id_type to, bool check = true) {
        check_external_not_reserved(&to, &to + 1);
        if (check) {
            if (!has_external(from)) {
                throw ANNEXCEPTION("Index does not contain external ID {}!", from);
            }
            if (has_external(to)) {
                throw ANNEXCEPTION("Index already contains external ID {}!", to);
            }
        }

        auto itr = external_to_internal_.find(from);
        auto internal = itr->second;

        // Updating the external-to-internal ID is easy.
        store_external(internal, to);
        external_to_internal_.erase(itr);
        external_to_internal_.insert({to, internal});
    }

    ///
    /// @brief Delete entries from internal IDs.
    ///
    /// @param internal_ids A container with the internal ids to delete. Must implement a
    ///     forward iterator. Furthermore, all entries must be unique (not checked).
    /// @param check Check if all ids actually exist in the translation table. This is only
    ///     safe to set to ``false`` if it can be guaranteed that all entries in
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
            store_external(i, invalid_external_id);
            external_to_internal_.erase(e);
        }
    }

    ///
    /// @brief Delete entries from external IDs.
    ///
    /// @param external_ids A container with the external ids to delete. Must implement a
    ///     forward iterator. Furthermore, all entries must be unique (not checked).
    /// @param check Check if all ids actually exist in the translation table. This is only
    ///     safe to set to ``false`` if it can be guaranteed that all entries in
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
            store_external(i, invalid_external_id);
            external_to_internal_.erase(e);
        }
    }

    ///
    /// @brief Ensure that **none** of the external ids in the iterator is the reserved
    ///     ``invalid_external_id``.
    ///
    /// @param begin Start forward iterator to the external IDs to check.
    /// @param end End forward iterator to the external IDs to check.
    ///
    template <class Begin, class End>
    void check_external_not_reserved(const Begin& begin, const End& end) const {
        check(
            begin,
            end,
            [](auto e) { return std::cmp_equal(e, invalid_external_id); },
            "Reserved external"
        );
    }

    ///
    /// @brief Ensure that **none** of the external ids in the iterator exist yet.
    ///
    /// @param begin Start forward iterator to the external IDs to check.
    /// @param end End forward iterator to the external IDs to check.
    ///
    template <class Begin, class End>
    void check_external_free(const Begin& begin, const End& end) const {
        check(
            begin,
            end,
            [this](auto e) { return has_external(e); },
            "Index already contains external"
        );
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
            [this](auto e) { return has_external(e); },
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
        check(
            begin,
            end,
            [this](auto i) { return has_internal(i); },
            "Index already contains internal"
        );
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
            [this](auto i) { return has_internal(i); },
            "Index does not contain internal",
            std::logical_not()
        );
    }

    ///// Saving and Loading
    static constexpr std::string_view kind = "external to internal id translation";
    static constexpr std::string_view serialization_schema =
        "external_to_internal_translation";
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);

    lib::SaveTable metadata() const {
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {{"kind", kind},
             {"num_points", lib::save(size())},
             {"external_id_type", lib::save(datatype_v<external_id_type>)},
             {"internal_id_type", lib::save(datatype_v<internal_id_type>)}}
        );
    }

    void save(std::ostream& os) const {
        for (auto i = begin(), iend = end(); i != iend; ++i) {
            // N.B.: Apparently `std::pair` of integers is not trivially copyable ...
            lib::write_binary(os, i->first);
            lib::write_binary(os, i->second);
        }
    }

    lib::SaveTable save(const lib::SaveContext& ctx) const {
        auto filename = ctx.generate_name("id_translation", "binary");
        // Save the translations to a file.
        auto os = lib::open_write(filename);
        save(os);
        auto table = metadata();
        table.insert("filename", lib::save(filename.filename()));
        return table;
    }

    static void validate(const lib::ContextFreeLoadTable& table) {
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
    }

    static IDTranslatorBase load(const lib::ContextFreeLoadTable& table, std::istream& is) {
        IDTranslatorBase::validate(table);
        auto num_points = lib::load_at<size_t>(table, "num_points");

        auto translator = IDTranslatorBase{};
        for (size_t i = 0; i < num_points; ++i) {
            auto external_id = lib::read_binary<external_id_type>(is);
            auto internal_id = lib::read_binary<internal_id_type>(is);
            translator.insert_translation(external_id, internal_id);
        }
        return translator;
    }

    static IDTranslatorBase load(const lib::LoadTable& table) {
        IDTranslatorBase::validate(table);

        // Now that we've more-or-less validated the metadata, time to start loading
        // the points.
        auto resolved = table.resolve_at("filename");
        auto is = lib::open_read(resolved);
        return IDTranslatorBase::load(table, is);
    }

  private:
    // The converting constructor takes the other variant's external->internal map.
    friend class IDTranslatorBase<!Concurrent>;

    using i_to_e_traits = VectorMapTraits<external_id_type, Concurrent>;

    template <class Begin, class End, class Contains, class Modifier = lib::identity>
    void check(
        const Begin& begin,
        const End& end,
        const Contains& contains,
        const char* message,
        Modifier modify = lib::identity()
    ) const {
        for (auto i = begin; i != end; ++i) {
            if (modify(contains(*i))) {
                throw ANNEXCEPTION("{} ID {}!", message, *i);
            }
        }
    }

    tsl::robin_map<external_id_type, internal_id_type> external_to_internal_{};
    // `typename` is required by compilers without P0634 (clang before 16).
    typename i_to_e_traits::vector_type internal_to_external_{};

    // Returns `invalid_external_id` for a slot that is unmapped or beyond the table.
    external_id_type load_external(internal_id_type i) const {
        return i_to_e_traits::load(internal_to_external_, size_t{i}, invalid_external_id);
    }

    // Writers only; grows the table with unmapped slots first when `i` is beyond it.
    void store_external(internal_id_type i, external_id_type e) {
        i_to_e_traits::store(internal_to_external_, size_t{i}, e, invalid_external_id);
    }
};
} // namespace detail

/// The translator for single-threaded use. See ``detail::IDTranslatorBase``.
using IDTranslator = detail::IDTranslatorBase<false>;
/// The translator whose internal-to-external lookups may run without a lock against one
/// writer. See ``detail::IDTranslatorBase``.
using ConcurrentIDTranslator = detail::IDTranslatorBase<true>;

} // namespace svs
