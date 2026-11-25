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
#include "svs/concepts/data.h"
#include "svs/concepts/vector_set.h"
#include "svs/core/data/simple.h"
#include "svs/core/data/vector_set_view.h"
#include "svs/lib/exception.h"
#include "svs/lib/file.h"
#include "svs/lib/memory.h"
#include "svs/lib/narrow.h"
#include "svs/lib/saveload.h"

// stl
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <utility>
#include <vector>

namespace svs {
namespace data {

///
/// @brief Owning dataset whose elements are vector sets: documents made of token
/// vectors, stored CSR-style exactly as ``ConstVectorSetView`` reads them.
///
/// The token matrix (every token vector of every document, concatenated row-major) is
/// held in a ``SimpleData<T, Extent, Alloc>``, so SVS's allocators (including the
/// hugepage allocator), alignment, and binary file format are reused as-is. A
/// ``std::vector<uint64_t>`` of length ``size() + 1`` holds the CSR offsets, counted in
/// token vectors.
///
/// @tparam T The element type of each token vector.
/// @tparam Extent The compile-time dimensionality shared by every token vector, or
///     ``svs::Dynamic`` to resolve it at run time. Default: ``svs::Dynamic``.
/// @tparam Alloc The allocator used for the token matrix. Must not be a ``Blocked``
///     allocator; see the ``static_assert`` below.
///
template <typename T, size_t Extent = Dynamic, typename Alloc = lib::Allocator<T>>
class VectorSetData {
    // A document's tokens must land in one contiguous run (`get_datum` returns a single
    // span into the token matrix); the Blocked allocator splits storage into separate
    // blocks, which would tear a document's tokens across blocks.
    static_assert(
        !is_blocked_v<Alloc>,
        "VectorSetData requires contiguous per-document token storage; a Blocked "
        "allocator is not supported."
    );

  public:
    /// The element type of each token vector.
    using element_type = T;
    /// The type returned by ``get_datum``: a constant handle to one document's tokens.
    using value_type = ConstVectorSet<T, Extent>;
    /// @copydoc value_type
    using const_value_type = ConstVectorSet<T, Extent>;
    /// The allocator type backing the token matrix.
    using allocator_type = Alloc;
    /// The concrete type of the token-matrix member.
    using tokens_type = SimpleData<T, Extent, Alloc>;
    // VamanaIndex reads Data::extent in its class body (index/vamana/index.h); without
    // it this type cannot be used as an index's dataset.
    static constexpr size_t extent = Extent;

    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
    static constexpr std::string_view serialization_schema = "vector_set_data";

    ///
    /// @brief Take ownership of an already-built token matrix and offsets array.
    ///
    /// @param tokens The row-major token matrix: ``total_tokens`` rows (the sum of every
    ///     document's cardinality) by ``dimensions()`` columns, where ``total_tokens``
    ///     is derived from ``offsets.back()``.
    /// @param offsets CSR-style offsets counted in token vectors: length
    ///     ``size() + 1``, non-decreasing, ``offsets.front() == 0``, and
    ///     ``offsets.back()`` equal to the number of rows in ``tokens``.
    ///
    /// Both arguments are consumed by move into this container; no payload copy is
    /// made. Pass ``std::move(tokens)`` and ``std::move(offsets)`` to hand over a
    /// caller's own storage. A caller that wants to keep an independent copy must copy
    /// explicitly before calling this constructor.
    ///
    VectorSetData(tokens_type tokens, std::vector<uint64_t> offsets)
        : tokens_{std::move(tokens)}
        , offsets_{std::move(offsets)} {
        detail::validate_vector_set_layout<Extent>(
            offsets_, tokens_.size() * tokens_.dimensions(), tokens_.dimensions()
        );
    }

    ///
    /// @brief Copy a borrowed view into a new, owned ``VectorSetData``.
    ///
    /// @param view The borrowed vector-set view to copy from.
    /// @param allocator The allocator used for the new token matrix. Default:
    ///     a default-constructed ``Alloc``.
    ///
    /// For the common case where a caller only has third-party, borrowed memory and
    /// wants SVS to own a copy. The copy is performed document by document, not in
    /// parallel; callers with large data who can hand over ownership instead should
    /// prefer the move constructor above.
    ///
    static VectorSetData
    copy_from(ConstVectorSetView<T, Extent> view, const Alloc& allocator = {}) {
        size_t dims = view.dimensions();
        auto tokens = tokens_type(view.total_tokens(), dims, allocator);
        auto offsets = std::vector<uint64_t>(view.size() + 1);
        size_t running = 0;
        for (size_t i = 0; i < view.size(); ++i) {
            offsets[i] = running;
            auto set = view.get_datum(i);
            for (size_t j = 0, jmax = set.cardinality(); j < jmax; ++j) {
                auto src = set.token(j);
                auto dst = tokens.get_datum(running + j);
                std::copy(src.begin(), src.end(), dst.begin());
            }
            running += set.cardinality();
        }
        offsets[view.size()] = running;
        return VectorSetData(std::move(tokens), std::move(offsets));
    }

    /// Return the number of documents (vector sets) in the dataset.
    size_t size() const { return offsets_.size() - 1; }
    /// Return the dimensionality shared by every token vector.
    size_t dimensions() const { return tokens_.dimensions(); }
    /// Return the number of token vectors in document ``i``.
    ///
    /// **Preconditions:** ``0 <= i < size()``.
    size_t cardinality(size_t i) const {
        assert(i < size());
        return lib::narrow<size_t>(offsets_[i + 1] - offsets_[i]);
    }
    /// Return the total number of token vectors across every document.
    size_t total_tokens() const { return tokens_.size(); }

    ///
    /// @brief Return a constant handle to the token vectors of document ``i``.
    ///
    /// **Preconditions:** ``0 <= i < size()``.
    ///
    const_value_type get_datum(size_t i) const {
        size_t set_cardinality = cardinality(i);
        size_t begin = lib::narrow<size_t>(offsets_[i]);
        size_t dims = dimensions();
        return ConstVectorSet<T, Extent>(
            std::span<const T>(tokens_.data() + begin * dims, set_cardinality * dims),
            set_cardinality,
            dims
        );
    }

    /// Prefetch the first token vector of document ``i``; see
    /// ``ConstVectorSetView::prefetch`` for why only the first token is prefetched.
    void prefetch(size_t i) const {
        if (cardinality(i) == 0) {
            return;
        }
        tokens_.prefetch(lib::narrow<size_t>(offsets_[i]));
    }

    /// Return the token matrix backing this dataset.
    const tokens_type& tokens() const { return tokens_; }
    /// Return the CSR offsets backing this dataset, counted in token vectors.
    std::span<const uint64_t> offsets() const { return offsets_; }

    ///
    /// @brief Return a non-owning ``ConstVectorSetView`` over this container's storage.
    ///
    /// The returned view is invalidated if this container is moved from or destroyed.
    ///
    ConstVectorSetView<T, Extent> view() const {
        return ConstVectorSetView<T, Extent>(
            std::span<const T>(tokens_.data(), tokens_.size() * dimensions()),
            offsets(),
            dimensions()
        );
    }

    ///// IO

    static bool check_load_compatibility(std::string_view schema, lib::Version version) {
        return schema == serialization_schema && version == save_version;
    }

    ///
    /// @brief Save this dataset to disk.
    ///
    /// The token matrix is saved by delegating to its own ``save``, nested as a
    /// sub-table under the ``"tokens"`` key. The offsets are written to a standalone
    /// binary file (raw ``uint64_t``, no header) named through ``ctx`` and referenced by
    /// the ``"offsets_file"``/``"num_offsets"`` metadata keys.
    ///
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        auto offsets_path = ctx.generate_name("vector_set_offsets", "bin");
        {
            auto io = lib::open_write(offsets_path);
            io.write(
                reinterpret_cast<const char*>(offsets_.data()),
                lib::narrow<std::streamsize>(offsets_.size() * sizeof(uint64_t))
            );
        }
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {{"tokens", lib::save(tokens_, ctx)},
             {"offsets_file", lib::save(offsets_path.filename())},
             {"num_offsets", lib::save(offsets_.size())}}
        );
    }

    ///
    /// @brief Reload a previously saved dataset.
    ///
    /// @param table The table containing saved hyper parameters, as produced by
    ///     ``save(const lib::SaveContext&)``.
    /// @param allocator The allocator used for the reloaded token matrix. Default: a
    ///     default-constructed ``Alloc``.
    ///
    /// Validation is performed by the constructor through the same shared layout check
    /// used by ``ConstVectorSetView``, so a corrupted offsets file or mismatched token
    /// count fails with the same diagnostics as an in-memory construction would.
    ///
    static VectorSetData load(const lib::LoadTable& table, const Alloc& allocator = {}) {
        auto tokens = lib::load_at<tokens_type>(table, "tokens", allocator);
        size_t num_offsets = lib::load_at<size_t>(table, "num_offsets");
        auto offsets_path = table.resolve_at("offsets_file");

        size_t expected_bytes = num_offsets * sizeof(uint64_t);
        size_t actual_bytes = std::filesystem::file_size(offsets_path);
        if (actual_bytes != expected_bytes) {
            throw ANNEXCEPTION(
                "Vector set offsets file {} has {} bytes but {} offsets ({} bytes) were "
                "expected",
                offsets_path,
                actual_bytes,
                num_offsets,
                expected_bytes
            );
        }

        auto offsets = std::vector<uint64_t>(num_offsets);
        {
            auto io = lib::open_read(offsets_path);
            io.read(
                reinterpret_cast<char*>(offsets.data()),
                lib::narrow<std::streamsize>(expected_bytes)
            );
        }
        return VectorSetData(std::move(tokens), std::move(offsets));
    }

    ///
    /// @brief Reload a dataset previously saved to ``path`` via
    /// ``lib::save_to_disk`` or ``VamanaIndex::save``.
    ///
    /// @param path The directory the dataset was saved into.
    /// @param allocator The allocator used for the reloaded token matrix. Default: a
    ///     default-constructed ``Alloc``.
    ///
    static VectorSetData
    load(const std::filesystem::path& path, const Alloc& allocator = {}) {
        return lib::load_from_disk<VectorSetData>(path, allocator);
    }

  private:
    tokens_type tokens_;
    std::vector<uint64_t> offsets_;
};

static_assert(ImmutableMemoryDataset<VectorSetData<float>>);
static_assert(ImmutableVectorSetDataset<VectorSetData<float>>);

} // namespace data

///
/// @brief Loader for owning vector-set datasets, mirroring ``svs::VectorDataLoader``.
///
/// @tparam T The element type of each token vector.
/// @tparam Extent The compile-time dimensionality of the token vectors. Default:
///     ``svs::Dynamic``.
/// @tparam Allocator The allocator to use for the memory backing the data when loaded.
///     Default: ``HugepageAllocator<T>``.
///
/// Satisfies the ``f.load()`` shape ``svs::detail::dispatch_load`` looks for, so an
/// instance may be passed anywhere a Vamana ``DataProto`` is expected, e.g. to
/// ``svs::index::vamana::auto_assemble``.
///
template <typename T, size_t Extent = Dynamic, typename Allocator = HugepageAllocator<T>>
class VectorSetDataLoader {
  public:
    /// @brief The full type of the loaded dataset.
    using return_type = data::VectorSetData<T, Extent, Allocator>;

    ///
    /// @brief Construct a new VectorSetDataLoader.
    ///
    /// @param path The directory previously used to save a ``VectorSetData`` (via
    ///     ``lib::save_to_disk`` or ``VamanaIndex::save``).
    /// @param allocator The allocator to use when loading. Default: a
    ///     default-constructed ``Allocator``.
    ///
    VectorSetDataLoader(const std::filesystem::path& path, const Allocator& allocator = {})
        : path_{path}
        , allocator_{allocator} {}

    /// @brief Load the dataset from disk.
    return_type load() const { return return_type::load(path_, allocator_); }

    ///// Members
    std::filesystem::path path_ = {};
    Allocator allocator_ = {};
};

} // namespace svs
