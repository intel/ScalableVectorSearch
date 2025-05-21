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

#define SVS_PACK_ARGS(...) __VA_ARGS__

#define SVS_CLASS_METHOD_MICROARCH_CASE(microarch, cls, method, args) \
    case svs::arch::MicroArch::microarch:                             \
        return cls<svs::arch::MicroArch::microarch>::method(args);    \
        break;

#define SVS_MICROARCH_FUNC_IF_SUPPORTED(uarch) SVS_MICROARCH_FUNC_##uarch(uarch)

// TODO: autogenerate this list
#if defined(__x86_64__)
// Macro used to apply `SVS_MICROARCH_FUNC` for each SUPPORTED uarch
// N.B.: do not forget to undefine `SVS_MICROARCH_FUNC` each time after use
#define SVS_FOR_EACH_MICROARCH                      \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(nehalem)        \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(westmere)       \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(sandybridge)    \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(ivybridge)      \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(haswell)        \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(broadwell)      \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(skylake)        \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(x86_64_v4)      \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(skylake_avx512) \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(cascadelake)    \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(cooperlake)     \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(icelake_client) \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(icelake_server) \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(sapphirerapids) \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(graniterapids)  \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(graniterapids_d)

// Macro used to apply `SVS_MICROARCH_FUNC` for each KNOWN uarch
#define SVS_FOR_EACH_KNOWN_MICROARCH   \
    SVS_MICROARCH_FUNC(nehalem)        \
    SVS_MICROARCH_FUNC(westmere)       \
    SVS_MICROARCH_FUNC(sandybridge)    \
    SVS_MICROARCH_FUNC(ivybridge)      \
    SVS_MICROARCH_FUNC(haswell)        \
    SVS_MICROARCH_FUNC(broadwell)      \
    SVS_MICROARCH_FUNC(skylake)        \
    SVS_MICROARCH_FUNC(x86_64_v4)      \
    SVS_MICROARCH_FUNC(skylake_avx512) \
    SVS_MICROARCH_FUNC(cascadelake)    \
    SVS_MICROARCH_FUNC(cooperlake)     \
    SVS_MICROARCH_FUNC(icelake_client) \
    SVS_MICROARCH_FUNC(icelake_server) \
    SVS_MICROARCH_FUNC(sapphirerapids) \
    SVS_MICROARCH_FUNC(graniterapids)  \
    SVS_MICROARCH_FUNC(graniterapids_d)

#if defined(SVS_MICROARCH_SUPPORT_nehalem)
#define SVS_MICROARCH_FUNC_nehalem(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_nehalem(uarch)
#endif

#if defined(SVS_MICROARCH_SUPPORT_westmere)
#define SVS_MICROARCH_FUNC_westmere(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_westmere(uarch)
#endif

#if defined(SVS_MICROARCH_SUPPORT_sandybridge)
#define SVS_MICROARCH_FUNC_sandybridge(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_sandybridge(uarch)
#endif

#if defined(SVS_MICROARCH_SUPPORT_ivybridge)
#define SVS_MICROARCH_FUNC_ivybridge(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_ivybridge(uarch)
#endif

#if defined(SVS_MICROARCH_SUPPORT_haswell)
#define SVS_MICROARCH_FUNC_haswell(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_haswell(uarch)
#endif

#if defined(SVS_MICROARCH_SUPPORT_broadwell)
#define SVS_MICROARCH_FUNC_broadwell(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_broadwell(uarch)
#endif

#if defined(SVS_MICROARCH_SUPPORT_skylake)
#define SVS_MICROARCH_FUNC_skylake(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_skylake(uarch)
#endif

#if defined(SVS_MICROARCH_SUPPORT_x86_64_v4)
#define SVS_MICROARCH_FUNC_x86_64_v4(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_x86_64_v4(uarch)
#endif

#if defined(SVS_MICROARCH_SUPPORT_skylake_avx512)
#define SVS_MICROARCH_FUNC_skylake_avx512(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_skylake_avx512(uarch)
#endif

#if defined(SVS_MICROARCH_SUPPORT_cascadelake)
#define SVS_MICROARCH_FUNC_cascadelake(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_cascadelake(uarch)
#endif

#if defined(SVS_MICROARCH_SUPPORT_cooperlake)
#define SVS_MICROARCH_FUNC_cooperlake(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_cooperlake(uarch)
#endif

#if defined(SVS_MICROARCH_SUPPORT_icelake_client)
#define SVS_MICROARCH_FUNC_icelake_client(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_icelake_client(uarch)
#endif

#if defined(SVS_MICROARCH_SUPPORT_icelake_server)
#define SVS_MICROARCH_FUNC_icelake_server(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_icelake_server(uarch)
#endif

#if defined(SVS_MICROARCH_SUPPORT_sapphirerapids)
#define SVS_MICROARCH_FUNC_sapphirerapids(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_sapphirerapids(uarch)
#endif

#if defined(SVS_MICROARCH_SUPPORT_graniterapids)
#define SVS_MICROARCH_FUNC_graniterapids(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_graniterapids(uarch)
#endif

#if defined(SVS_MICROARCH_SUPPORT_graniterapids_d)
#define SVS_MICROARCH_FUNC_graniterapids_d(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_graniterapids_d(uarch)
#endif

#elif defined(__aarch64__)

#if defined(__APPLE__)

#define SVS_FOR_EACH_MICROARCH          \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(m1) \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(m2)

#define SVS_FOR_EACH_KNOWN_MICROARCH \
    SVS_MICROARCH_FUNC(m1)           \
    SVS_MICROARCH_FUNC(m2)

#if defined(SVS_MICROARCH_SUPPORT_m1)
#define SVS_MICROARCH_FUNC_m1(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_m1(uarch)
#endif

#if defined(SVS_MICROARCH_SUPPORT_m2)
#define SVS_MICROARCH_FUNC_m2(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_m2(uarch)
#endif

#else

#define SVS_FOR_EACH_MICROARCH                   \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(neoverse_v1) \
    SVS_MICROARCH_FUNC_IF_SUPPORTED(neoverse_n2)

#define SVS_FOR_EACH_KNOWN_MICROARCH \
    SVS_MICROARCH_FUNC(neoverse_v1)  \
    SVS_MICROARCH_FUNC(neoverse_n2)

#if defined(SVS_MICROARCH_SUPPORT_neoverse_v1)
#define SVS_MICROARCH_FUNC_neoverse_v1(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_neoverse_v1(uarch)
#endif

#if defined(SVS_MICROARCH_SUPPORT_neoverse_n2)
#define SVS_MICROARCH_FUNC_neoverse_n2(uarch) SVS_MICROARCH_FUNC(uarch)
#else
#define SVS_MICROARCH_FUNC_neoverse_n2(uarch)
#endif

#endif
#endif
