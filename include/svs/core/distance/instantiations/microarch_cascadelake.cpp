#include "svs/core/distance/cosine.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/inner_product.h"
#include "svs/lib/arch.h"

namespace svs::distance {
template struct IP<svs::arch::MicroArch::cascadelake>;
template class L2<svs::arch::MicroArch::cascadelake>;
template class CosineSimilarity<svs::arch::MicroArch::cascadelake>;
} // namespace svs::distance
