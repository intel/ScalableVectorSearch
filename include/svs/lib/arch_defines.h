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
#define SVS_TARGET_MICROARCH svs::arch::MicroArch::SVS_TUNE_TARGET

#define SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(           \
    microarch, return_type, cls, method, template_args, args \
)                                                            \
    extern template return_type                              \
        cls<svs::arch::MicroArch::microarch>::method<template_args>(args);

// TODO: autogenerate this list
#if defined(__x86_64__)

#if defined(SVS_MICROARCH_SUPPORT_nehalem)
#define SVS_MICROARCH_COMPILED_nehalem MicroArch::nehalem,
#define SVS_CLASS_METHOD_MICROARCH_CASE_nehalem(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(nehalem, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_nehalem(return_type, cls, method, template_args, args) \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(                                         \
        nehalem,                                                                       \
        return_type,                                                                   \
        cls,                                                                           \
        method,                                                                        \
        SVS_PACK_ARGS(template_args),                                                  \
        SVS_PACK_ARGS(args)                                                            \
    )
#else
#define SVS_MICROARCH_COMPILED_nehalem
#define SVS_CLASS_METHOD_MICROARCH_CASE_nehalem(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_nehalem(return_type, cls, method, template_args, args)
#endif

#if defined(SVS_MICROARCH_SUPPORT_westmere)
#define SVS_MICROARCH_COMPILED_westmere MicroArch::westmere,
#define SVS_CLASS_METHOD_MICROARCH_CASE_westmere(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(westmere, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_westmere(return_type, cls, method, template_args, args) \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(                                          \
        westmere,                                                                       \
        return_type,                                                                    \
        cls,                                                                            \
        method,                                                                         \
        SVS_PACK_ARGS(template_args),                                                   \
        SVS_PACK_ARGS(args)                                                             \
    )
#else
#define SVS_MICROARCH_COMPILED_westmere
#define SVS_CLASS_METHOD_MICROARCH_CASE_westmere(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_westmere(return_type, cls, method, template_args, args)
#endif

#if defined(SVS_MICROARCH_SUPPORT_sandybridge)
#define SVS_MICROARCH_COMPILED_sandybridge MicroArch::sandybridge,
#define SVS_CLASS_METHOD_MICROARCH_CASE_sandybridge(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(sandybridge, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_sandybridge(return_type, cls, method, template_args, args) \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(                                             \
        sandybridge,                                                                       \
        return_type,                                                                       \
        cls,                                                                               \
        method,                                                                            \
        SVS_PACK_ARGS(template_args),                                                      \
        SVS_PACK_ARGS(args)                                                                \
    )
#else
#define SVS_MICROARCH_COMPILED_sandybridge
#define SVS_CLASS_METHOD_MICROARCH_CASE_sandybridge(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_sandybridge(return_type, cls, method, template_args, args)
#endif

#if defined(SVS_MICROARCH_SUPPORT_ivybridge)
#define SVS_MICROARCH_COMPILED_ivybridge MicroArch::ivybridge,
#define SVS_CLASS_METHOD_MICROARCH_CASE_ivybridge(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(ivybridge, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_ivybridge(return_type, cls, method, template_args, args) \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(                                           \
        ivybridge,                                                                       \
        return_type,                                                                     \
        cls,                                                                             \
        method,                                                                          \
        SVS_PACK_ARGS(template_args),                                                    \
        SVS_PACK_ARGS(args)                                                              \
    )
#else
#define SVS_MICROARCH_COMPILED_ivybridge
#define SVS_CLASS_METHOD_MICROARCH_CASE_ivybridge(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_ivybridge(return_type, cls, method, template_args, args)
#endif

#if defined(SVS_MICROARCH_SUPPORT_haswell)
#define SVS_MICROARCH_COMPILED_haswell MicroArch::haswell,
#define SVS_CLASS_METHOD_MICROARCH_CASE_haswell(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(haswell, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_haswell(return_type, cls, method, template_args, args) \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(                                         \
        haswell,                                                                       \
        return_type,                                                                   \
        cls,                                                                           \
        method,                                                                        \
        SVS_PACK_ARGS(template_args),                                                  \
        SVS_PACK_ARGS(args)                                                            \
    )
#else
#define SVS_MICROARCH_COMPILED_haswell
#define SVS_CLASS_METHOD_MICROARCH_CASE_haswell(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_haswell(return_type, cls, method, template_args, args)
#endif

#if defined(SVS_MICROARCH_SUPPORT_broadwell)
#define SVS_MICROARCH_COMPILED_broadwell MicroArch::broadwell,
#define SVS_CLASS_METHOD_MICROARCH_CASE_broadwell(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(broadwell, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_broadwell(return_type, cls, method, template_args, args) \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(                                           \
        broadwell,                                                                       \
        return_type,                                                                     \
        cls,                                                                             \
        method,                                                                          \
        SVS_PACK_ARGS(template_args),                                                    \
        SVS_PACK_ARGS(args)                                                              \
    )
#else
#define SVS_MICROARCH_COMPILED_broadwell
#define SVS_CLASS_METHOD_MICROARCH_CASE_broadwell(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_broadwell(return_type, cls, method, template_args, args)
#endif

#if defined(SVS_MICROARCH_SUPPORT_skylake)
#define SVS_MICROARCH_COMPILED_skylake MicroArch::skylake,
#define SVS_CLASS_METHOD_MICROARCH_CASE_skylake(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(skylake, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_skylake(return_type, cls, method, template_args, args) \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(                                         \
        skylake,                                                                       \
        return_type,                                                                   \
        cls,                                                                           \
        method,                                                                        \
        SVS_PACK_ARGS(template_args),                                                  \
        SVS_PACK_ARGS(args)                                                            \
    )
#else
#define SVS_MICROARCH_COMPILED_skylake
#define SVS_CLASS_METHOD_MICROARCH_CASE_skylake(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_skylake(return_type, cls, method, template_args, args)
#endif

#if defined(SVS_MICROARCH_SUPPORT_x86_64_v4)
#define SVS_MICROARCH_COMPILED_x86_64_v4 MicroArch::x86_64_v4,
#define SVS_CLASS_METHOD_MICROARCH_CASE_x86_64_v4(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(x86_64_v4, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_x86_64_v4(return_type, cls, method, template_args, args) \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(                                           \
        x86_64_v4,                                                                       \
        return_type,                                                                     \
        cls,                                                                             \
        method,                                                                          \
        SVS_PACK_ARGS(template_args),                                                    \
        SVS_PACK_ARGS(args)                                                              \
    )
#else
#define SVS_MICROARCH_COMPILED_x86_64_v4
#define SVS_CLASS_METHOD_MICROARCH_CASE_x86_64_v4(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_x86_64_v4(return_type, cls, method, template_args, args)
#endif

#if defined(SVS_MICROARCH_SUPPORT_skylake_avx512)
#define SVS_MICROARCH_COMPILED_skylake_avx512 MicroArch::skylake_avx512,
#define SVS_CLASS_METHOD_MICROARCH_CASE_skylake_avx512(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(skylake_avx512, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_skylake_avx512(   \
    return_type, cls, method, template_args, args \
)                                                 \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(    \
        skylake_avx512,                           \
        return_type,                              \
        cls,                                      \
        method,                                   \
        SVS_PACK_ARGS(template_args),             \
        SVS_PACK_ARGS(args)                       \
    )
#else
#define SVS_MICROARCH_COMPILED_skylake_avx512
#define SVS_CLASS_METHOD_MICROARCH_CASE_skylake_avx512(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_skylake_avx512(   \
    return_type, cls, method, template_args, args \
)
#endif

#if defined(SVS_MICROARCH_SUPPORT_cascadelake)
#define SVS_MICROARCH_COMPILED_cascadelake MicroArch::cascadelake,
#define SVS_CLASS_METHOD_MICROARCH_CASE_cascadelake(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(cascadelake, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_cascadelake(return_type, cls, method, template_args, args) \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(                                             \
        cascadelake,                                                                       \
        return_type,                                                                       \
        cls,                                                                               \
        method,                                                                            \
        SVS_PACK_ARGS(template_args),                                                      \
        SVS_PACK_ARGS(args)                                                                \
    )
#else
#define SVS_MICROARCH_COMPILED_cascadelake
#define SVS_CLASS_METHOD_MICROARCH_CASE_cascadelake(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_cascadelake(return_type, cls, method, template_args, args)
#endif

#if defined(SVS_MICROARCH_SUPPORT_cooperlake)
#define SVS_MICROARCH_COMPILED_cooperlake MicroArch::cooperlake,
#define SVS_CLASS_METHOD_MICROARCH_CASE_cooperlake(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(cooperlake, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_cooperlake(return_type, cls, method, template_args, args) \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(                                            \
        cooperlake,                                                                       \
        return_type,                                                                      \
        cls,                                                                              \
        method,                                                                           \
        SVS_PACK_ARGS(template_args),                                                     \
        SVS_PACK_ARGS(args)                                                               \
    )
#else
#define SVS_MICROARCH_COMPILED_cooperlake
#define SVS_CLASS_METHOD_MICROARCH_CASE_cooperlake(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_cooperlake(return_type, cls, method, template_args, args)
#endif

#if defined(SVS_MICROARCH_SUPPORT_icelake_client)
#define SVS_MICROARCH_COMPILED_icelake_client MicroArch::icelake_client,
#define SVS_CLASS_METHOD_MICROARCH_CASE_icelake_client(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(icelake_client, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_icelake_client(   \
    return_type, cls, method, template_args, args \
)                                                 \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(    \
        icelake_client,                           \
        return_type,                              \
        cls,                                      \
        method,                                   \
        SVS_PACK_ARGS(template_args),             \
        SVS_PACK_ARGS(args)                       \
    )
#else
#define SVS_MICROARCH_COMPILED_icelake_client
#define SVS_CLASS_METHOD_MICROARCH_CASE_icelake_client(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_icelake_client(   \
    return_type, cls, method, template_args, args \
)
#endif

#if defined(SVS_MICROARCH_SUPPORT_icelake_server)
#define SVS_MICROARCH_COMPILED_icelake_server MicroArch::icelake_server,
#define SVS_CLASS_METHOD_MICROARCH_CASE_icelake_server(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(icelake_server, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_icelake_server(   \
    return_type, cls, method, template_args, args \
)                                                 \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(    \
        icelake_server,                           \
        return_type,                              \
        cls,                                      \
        method,                                   \
        SVS_PACK_ARGS(template_args),             \
        SVS_PACK_ARGS(args)                       \
    )
#else
#define SVS_MICROARCH_COMPILED_icelake_server
#define SVS_CLASS_METHOD_MICROARCH_CASE_icelake_server(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_icelake_server(   \
    return_type, cls, method, template_args, args \
)
#endif

#if defined(SVS_MICROARCH_SUPPORT_sapphirerapids)
#define SVS_MICROARCH_COMPILED_sapphirerapids MicroArch::sapphirerapids,
#define SVS_CLASS_METHOD_MICROARCH_CASE_sapphirerapids(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(sapphirerapids, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_sapphirerapids(   \
    return_type, cls, method, template_args, args \
)                                                 \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(    \
        sapphirerapids,                           \
        return_type,                              \
        cls,                                      \
        method,                                   \
        SVS_PACK_ARGS(template_args),             \
        SVS_PACK_ARGS(args)                       \
    )
#else
#define SVS_MICROARCH_COMPILED_sapphirerapids
#define SVS_CLASS_METHOD_MICROARCH_CASE_sapphirerapids(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_sapphirerapids(   \
    return_type, cls, method, template_args, args \
)
#endif

#if defined(SVS_MICROARCH_SUPPORT_graniterapids)
#define SVS_MICROARCH_COMPILED_graniterapids MicroArch::graniterapids,
#define SVS_CLASS_METHOD_MICROARCH_CASE_graniterapids(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(graniterapids, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_graniterapids(    \
    return_type, cls, method, template_args, args \
)                                                 \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(    \
        graniterapids,                            \
        return_type,                              \
        cls,                                      \
        method,                                   \
        SVS_PACK_ARGS(template_args),             \
        SVS_PACK_ARGS(args)                       \
    )
#else
#define SVS_MICROARCH_COMPILED_graniterapids
#define SVS_CLASS_METHOD_MICROARCH_CASE_graniterapids(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_graniterapids(return_type, cls, method, template_args, args)
#endif

#if defined(SVS_MICROARCH_SUPPORT_graniterapids_d)
#define SVS_MICROARCH_COMPILED_graniterapids_d MicroArch::graniterapids_d,
#define SVS_CLASS_METHOD_MICROARCH_CASE_graniterapids_d(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(graniterapids_d, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_graniterapids_d(  \
    return_type, cls, method, template_args, args \
)                                                 \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(    \
        graniterapids_d,                          \
        return_type,                              \
        cls,                                      \
        method,                                   \
        SVS_PACK_ARGS(template_args),             \
        SVS_PACK_ARGS(args)                       \
    )
#else
#define SVS_MICROARCH_COMPILED_graniterapids_d
#define SVS_CLASS_METHOD_MICROARCH_CASE_graniterapids_d(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_graniterapids_d(  \
    return_type, cls, method, template_args, args \
)
#endif

#elif defined(__aarch64__)

#if defined(__APPLE__)

#if defined(SVS_MICROARCH_SUPPORT_m1)
#define SVS_MICROARCH_COMPILED_m1 MicroArch::m1,
#define SVS_CLASS_METHOD_MICROARCH_CASE_m1(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(m1, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_m1(return_type, cls, method, template_args, args)       \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(                                          \
        m1, return_type, cls, method, SVS_PACK_ARGS(template_args), SVS_PACK_ARGS(args) \
    )
#else
#define SVS_MICROARCH_COMPILED_m1
#define SVS_CLASS_METHOD_MICROARCH_CASE_m1(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_m1(return_type, cls, method, template_args, args)
#endif

#if defined(SVS_MICROARCH_SUPPORT_m2)
#define SVS_MICROARCH_COMPILED_m2 MicroArch::m2,
#define SVS_CLASS_METHOD_MICROARCH_CASE_m2(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(m2, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_m2(return_type, cls, method, template_args, args)       \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(                                          \
        m2, return_type, cls, method, SVS_PACK_ARGS(template_args), SVS_PACK_ARGS(args) \
    )
#else
#define SVS_MICROARCH_COMPILED_m2
#define SVS_CLASS_METHOD_MICROARCH_CASE_m2(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_m2(return_type, cls, method, template_args, args)
#endif

#else

#if defined(SVS_MICROARCH_SUPPORT_neoverse_v1)
#define SVS_MICROARCH_COMPILED_neoverse_v1 MicroArch::neoverse_v1,
#define SVS_CLASS_METHOD_MICROARCH_CASE_neoverse_v1(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(neoverse_v1, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_neoverse_v1(return_type, cls, method, template_args, args) \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(                                             \
        neoverse_v1,                                                                       \
        return_type,                                                                       \
        cls,                                                                               \
        method,                                                                            \
        SVS_PACK_ARGS(template_args),                                                      \
        SVS_PACK_ARGS(args)                                                                \
    )
#else
#define SVS_MICROARCH_COMPILED_neoverse_v1
#define SVS_CLASS_METHOD_MICROARCH_CASE_neoverse_v1(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_neoverse_v1(return_type, cls, method, template_args, args)
#endif

#if defined(SVS_MICROARCH_SUPPORT_neoverse_n2)
#define SVS_MICROARCH_COMPILED_neoverse_n2 MicroArch::neoverse_n2,
#define SVS_CLASS_METHOD_MICROARCH_CASE_neoverse_n2(cls, method, args) \
    SVS_CLASS_METHOD_MICROARCH_CASE(neoverse_n2, cls, method, SVS_PACK_ARGS(args))
#define SVS_EXTERN_CLASS_METHOD_neoverse_n2(return_type, cls, method, template_args, args) \
    SVS_EXTERN_CLASS_METHOD_TMPL_BY_MICROARCH(                                             \
        neoverse_n2,                                                                       \
        return_type,                                                                       \
        cls,                                                                               \
        method,                                                                            \
        SVS_PACK_ARGS(template_args),                                                      \
        SVS_PACK_ARGS(args)                                                                \
    )
#else
#define SVS_MICROARCH_COMPILED_neoverse_n2
#define SVS_CLASS_METHOD_MICROARCH_CASE_neoverse_n2(cls, method, args)
#define SVS_EXTERN_CLASS_METHOD_neoverse_n2(return_type, cls, method, template_args, args)
#endif

#endif
#endif
