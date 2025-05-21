#include "svs/core/distance/cosine.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/inner_product.h"
#include "svs/lib/arch.h"

namespace svs::distance {
extern template struct IP<svs::arch::MicroArch::baseline>;

#if defined(__x86_64__)
extern template struct IP<svs::arch::MicroArch::x86_64_v2>;
extern template struct IP<svs::arch::MicroArch::nehalem>;
extern template struct IP<svs::arch::MicroArch::westmere>;
extern template struct IP<svs::arch::MicroArch::sandybridge>;
extern template struct IP<svs::arch::MicroArch::ivybridge>;
extern template struct IP<svs::arch::MicroArch::haswell>;
extern template struct IP<svs::arch::MicroArch::broadwell>;
extern template struct IP<svs::arch::MicroArch::skylake>;
extern template struct IP<svs::arch::MicroArch::x86_64_v4>;
extern template struct IP<svs::arch::MicroArch::skylake_avx512>;
extern template struct IP<svs::arch::MicroArch::cascadelake>;
extern template struct IP<svs::arch::MicroArch::cooperlake>;
extern template struct IP<svs::arch::MicroArch::icelake_client>;
extern template struct IP<svs::arch::MicroArch::icelake_server>;
extern template struct IP<svs::arch::MicroArch::sapphirerapids>;
extern template struct IP<svs::arch::MicroArch::graniterapids>;
extern template struct IP<svs::arch::MicroArch::graniterapids_d>;
#elif defined(__aarch64__)
#if defined(__APPLE__)
extern template struct IP<svs::arch::MicroArch::m1>;
extern template struct IP<svs::arch::MicroArch::m2>;
#else
extern template struct IP<svs::arch::MicroArch::neoverse_v1>;
extern template struct IP<svs::arch::MicroArch::neoverse_n2>;
#endif // __APPLE__
#endif // __aarch64__

#if defined(__x86_64__)
extern template class L2<svs::arch::MicroArch::x86_64_v2>;
extern template class L2<svs::arch::MicroArch::nehalem>;
extern template class L2<svs::arch::MicroArch::westmere>;
extern template class L2<svs::arch::MicroArch::sandybridge>;
extern template class L2<svs::arch::MicroArch::ivybridge>;
extern template class L2<svs::arch::MicroArch::haswell>;
extern template class L2<svs::arch::MicroArch::broadwell>;
extern template class L2<svs::arch::MicroArch::skylake>;
extern template class L2<svs::arch::MicroArch::x86_64_v4>;
extern template class L2<svs::arch::MicroArch::skylake_avx512>;
extern template class L2<svs::arch::MicroArch::cascadelake>;
extern template class L2<svs::arch::MicroArch::cooperlake>;
extern template class L2<svs::arch::MicroArch::icelake_client>;
extern template class L2<svs::arch::MicroArch::icelake_server>;
extern template class L2<svs::arch::MicroArch::sapphirerapids>;
extern template class L2<svs::arch::MicroArch::graniterapids>;
extern template class L2<svs::arch::MicroArch::graniterapids_d>;
#elif defined(__aarch64__)
#if defined(__APPLE__)
extern template class L2<svs::arch::MicroArch::m1>;
extern template class L2<svs::arch::MicroArch::m2>;
#else
extern template class L2<svs::arch::MicroArch::neoverse_v1>;
extern template class L2<svs::arch::MicroArch::neoverse_n2>;
#endif // __APPLE__
#endif // __aarch64__

#if defined(__x86_64__)
extern template class CosineSimilarity<svs::arch::MicroArch::x86_64_v2>;
extern template class CosineSimilarity<svs::arch::MicroArch::nehalem>;
extern template class CosineSimilarity<svs::arch::MicroArch::westmere>;
extern template class CosineSimilarity<svs::arch::MicroArch::sandybridge>;
extern template class CosineSimilarity<svs::arch::MicroArch::ivybridge>;
extern template class CosineSimilarity<svs::arch::MicroArch::haswell>;
extern template class CosineSimilarity<svs::arch::MicroArch::broadwell>;
extern template class CosineSimilarity<svs::arch::MicroArch::skylake>;
extern template class CosineSimilarity<svs::arch::MicroArch::x86_64_v4>;
extern template class CosineSimilarity<svs::arch::MicroArch::skylake_avx512>;
extern template class CosineSimilarity<svs::arch::MicroArch::cascadelake>;
extern template class CosineSimilarity<svs::arch::MicroArch::cooperlake>;
extern template class CosineSimilarity<svs::arch::MicroArch::icelake_client>;
extern template class CosineSimilarity<svs::arch::MicroArch::icelake_server>;
extern template class CosineSimilarity<svs::arch::MicroArch::sapphirerapids>;
extern template class CosineSimilarity<svs::arch::MicroArch::graniterapids>;
extern template class CosineSimilarity<svs::arch::MicroArch::graniterapids_d>;
#elif defined(__aarch64__)
#if defined(__APPLE__)
extern template class CosineSimilarity<svs::arch::MicroArch::m1>;
extern template class CosineSimilarity<svs::arch::MicroArch::m2>;
#else
extern template class CosineSimilarity<svs::arch::MicroArch::neoverse_v1>;
extern template class CosineSimilarity<svs::arch::MicroArch::neoverse_n2>;
#endif // __APPLE__
#endif // __aarch64__

} // namespace svs::distance
