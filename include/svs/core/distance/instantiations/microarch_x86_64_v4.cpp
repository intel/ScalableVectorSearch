#include "svs/core/distance/cosine.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/inner_product.h"
#include "svs/lib/arch.h"

namespace svs::distance {
template struct IP<svs::arch::MicroArch::x86_64_v4>;
template class L2<svs::arch::MicroArch::x86_64_v4>;
template class CosineSimilarity<svs::arch::MicroArch::x86_64_v4>;
} // namespace svs::distance
