/*
 * Copyright 2024 Intel Corporation
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

#include "svs/core/data/simple.h"
#include "svs/core/data/view.h"
#include "svs/lib/invoke.h"
#include "svs/lib/misc.h"
#include "svs/lib/readwrite.h"
#include "svs/lib/static.h"

#include "svs/core/data/simple.h"

// stl
#include <ostream>

namespace svs::index::inverted {

// // Define serialization for SimpleData.
// template <typename T, size_t N> struct VectorSerializer {
//     static_assert(std::is_trivially_copyable_v<T>);
//
//   public:
//     // Type Aliases
//     using type = T;
//     // Align data to the beginning of a cacheline.
//     static constexpr size_t global_alignment = 64;
//
//   public:
//     explicit VectorSerializer(lib::MaybeStatic<N> dimensions)
//         : dimensions_{dimensions} {}
//
//     // Serializer Interface.
//     template <typename Alloc>
//     size_t
//     write(std::ostream& stream, const data::SimpleData<T, N, Alloc>& data, size_t i)
//     const {
//         return lib::write_binary(stream, data.get_datum(i));
//     }
//
//     data::SimpleDataView<T, N> view(void* ptr, size_t num_points) const {
//         return data::SimpleDataView<T, N>(
//             reinterpret_cast<T*>(ptr), num_points, dimensions_
//         );
//     }
//
//     ///// Saving and Loading.
//     // N.B.: Keep staticly sized serializers compatible with dynamically sized ones.
//     static constexpr lib::Version save_version{0, 0, 0};
//     lib::SaveTable save() const {
//         return lib::SaveTable(save_version, {SVS_LIST_SAVE_(dimensions)});
//     }
//
//     static VectorSerializer load(const toml::table& table, const lib::Version& version) {
//         if (version != save_version) {
//             throw ANNEXCEPTION("Version mismatch!");
//         }
//
//         return VectorSerializer(lib::MaybeStatic<N>(SVS_LOAD_MEMBER_AT_(table,
//         dimensions))
//         );
//     }
//
//   public:
//     size_t dimensions_;
// };

namespace extensions {

/////
///// Clustering
/////

// Adapt distance for pruning.
struct ClusteringPruneCPO {
    using This = ClusteringPruneCPO;

    template <typename Data, typename Distance>
    auto operator()(const Data& data, const Distance& distance) const {
        // Did someone extend this?
        if constexpr (svs::svs_invocable<This, const Data&, const Distance&>) {
            return svs::svs_invoke(*this, data, distance);
        } else {
            // Return the original distance unchanged.
            return distance;
        }
    }
};

inline constexpr ClusteringPruneCPO clustering_distance{};

struct PrepareIndexSearchCPO {
    using This = PrepareIndexSearchCPO;

    template <typename Original, std::integral I>
    using return_type = svs::svs_invoke_result_t<This, const Original&, std::span<const I>>;

    template <typename Original, std::integral I>
    return_type<Original, I>
    operator()(const Original& original, std::span<const I> indices) const {
        return svs::svs_invoke(*this, original, indices);
    }
};

inline constexpr PrepareIndexSearchCPO prepare_index_search{};

/////
///// In-Memory extensions.
/////

// The following templates use `size` parameters because there are several repetitious
// instantiations of these objects.
//
// Each instantiation needs its own type for extensions, which is disambiguated by the
// value parameter.
template <size_t> struct CreationCPO {
    using This = CreationCPO;

    template <typename Data, typename Alloc>
    using return_type = svs::svs_invoke_result_t<This, const Data&, size_t, const Alloc&>;

    template <typename Data, typename Alloc>
    return_type<Data, Alloc>
    operator()(const Data& original, size_t new_size, const Alloc& allocator) const {
        return svs::svs_invoke(*this, original, new_size, allocator);
    }
};

inline constexpr CreationCPO<0> create_auxiliary_dataset{};
inline constexpr CreationCPO<1> create_first_level_dataset{};
inline constexpr CreationCPO<2> create_sparse_cluster{};
inline constexpr CreationCPO<3> create_dense_cluster{};

/////
///// Default Implementations
/////

template <typename T, size_t Extent, typename Alloc, std::integral I>
data::ConstDataView<data::SimpleData<T, Extent, Alloc>, std::span<const I>> svs_invoke(
    svs::tag_t<prepare_index_search>,
    const data::SimpleData<T, Extent, Alloc>& original,
    std::span<const I> indices
) {
    return data::make_const_view(original, indices);
}

// Default Implementation.
// TODO: Repition!!
template <typename T, size_t Extent, typename Alloc, typename NewAlloc>
svs::data::SimpleData<T, Extent, Alloc> svs_invoke(
    svs::tag_t<create_auxiliary_dataset>,
    const svs::data::SimpleData<T, Extent, Alloc>& original,
    size_t new_size,
    const NewAlloc& allocator
) {
    return svs::data::SimpleData<T, Extent, lib::rebind_allocator_t<T, NewAlloc>>(
        new_size, original.dimensions(), lib::rebind_allocator<T>(allocator)
    );
}

template <typename T, size_t Extent, typename Alloc, typename NewAlloc>
svs::data::SimpleData<T, Extent, lib::rebind_allocator_t<T, NewAlloc>> svs_invoke(
    svs::tag_t<create_first_level_dataset>,
    const svs::data::SimpleData<T, Extent, Alloc>& original,
    size_t new_size,
    const NewAlloc& allocator
) {
    return svs::data::SimpleData<T, Extent, lib::rebind_allocator_t<T, NewAlloc>>(
        new_size, original.dimensions(), lib::rebind_allocator<T>(allocator)
    );
}

template <typename T, size_t Extent, typename Alloc, typename NewAlloc>
svs::data::SimpleData<T, Extent, lib::rebind_allocator_t<T, NewAlloc>> svs_invoke(
    svs::tag_t<create_sparse_cluster>,
    const svs::data::SimpleData<T, Extent, Alloc>& original,
    size_t new_size,
    const NewAlloc& allocator
) {
    return svs::data::SimpleData<T, Extent, lib::rebind_allocator_t<T, NewAlloc>>(
        new_size, original.dimensions(), lib::rebind_allocator<T>(allocator)
    );
}

template <typename T, size_t Extent, typename Alloc, typename NewAlloc>
svs::data::SimpleData<T, Extent> svs_invoke(
    svs::tag_t<create_dense_cluster>,
    const svs::data::SimpleData<T, Extent, Alloc>& original,
    size_t new_size,
    // TODO: figure out how to better support custom allocator types in the dense case.
    const NewAlloc& SVS_UNUSED(allocator)
) {
    return svs::data::SimpleData<T, Extent>(new_size, original.dimensions());
}

// /////
// ///// SSD Extensions
// /////
//
// struct SerializerCPO {
//     template <data::ImmutableMemoryDataset Data>
//     using return_type = svs::svs_invoke_result_t<SerializerCPO, const Data&>;
//
//     template <data::ImmutableMemoryDataset Data>
//     return_type<Data> operator()(const Data& data) const {
//         return svs::svs_invoke(*this, data);
//     }
// };
//
// inline constexpr SerializerCPO serializer{};
//
// // CPO extension.
// template <typename T, size_t N, typename Alloc>
// VectorSerializer<T, N>
// svs_invoke(svs::tag_t<extensions::serializer>, const data::SimpleData<T, N, Alloc>& data)
// {
//     return VectorSerializer<T, N>(lib::MaybeStatic<N>(data.dimensions()));
// }

} // namespace extensions

} // namespace svs::index::inverted
