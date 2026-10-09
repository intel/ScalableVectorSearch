#if defined(__x86_64__)
#include "svs/core/distance/cosine.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/inner_product.h"

// This translation unit is compiled with -mavx512fp16 -mavx512vl by cmake to
// validate compilation of native fp16 intrinsics and provide an experimental
// native-FP16 implementation for Float16 x Float16 inner product. The
// implementation intentionally stays small and uses the existing SIMD helper
// infrastructure. When the compiler/CPU lacks FP16 support this TU still
// compiles (the CMake logic will omit the flags) and the code falls back to
// the AVX512 fp32-accumulating path.

namespace svs::distance {

/////
///// Native AVX512-FP16 kernel for Float16 x Float16 inner product.
/////
// IPNativeHalfOp32 uses the _Float16 / __m512h intrinsics introduced in
// Intel's AVX512-FP16 ISA extension (Sapphire Rapids and later). Each 512-bit
// register holds 32 fp16 elements, so simd_width=32. The masked load is
// implemented via _mm512_maskz_loadu_epi16 (zero-masking) cast to __m512h,
// which avoids the need for a separate blend and keeps inactive lanes at zero.
//
// Guard: __AVX512FP16__ is defined by GCC/Clang when -mavx512fp16 is passed.
// We do NOT use SVS_AVX512_F here because that only tests for AVX512F (present
// on Skylake) whereas FP16 native arithmetic requires Sapphire Rapids+.

#if defined(__AVX512FP16__)
struct IPNativeHalfOp32 {
    static constexpr size_t simd_width = 32;
    using mask_t = svs::mask_repr_t<32>;  // uint32_t

    static __m512h init() { return _mm512_setzero_ph(); }

    static __m512h load_a(const Float16* p) {
        return _mm512_loadu_ph(reinterpret_cast<const _Float16*>(p));
    }
    static __m512h load_a(mask_t m, const Float16* p) {
        return _mm512_castsi512_ph(
            _mm512_maskz_loadu_epi16((__mmask32)m, reinterpret_cast<const void*>(p)));
    }
    static __m512h load_b(const Float16* p) { return load_a(p); }
    static __m512h load_b(mask_t m, const Float16* p) { return load_a(m, p); }

    static __m512h accumulate(__m512h acc, __m512h a, __m512h b) {
        return _mm512_fmadd_ph(a, b, acc);
    }
    static __m512h accumulate(mask_t m, __m512h acc, __m512h a, __m512h b) {
        // Masked FMA: result[i] = mask[i] ? a[i]*b[i]+acc[i] : acc[i]
        return _mm512_mask_add_ph(acc, (__mmask32)m, acc, _mm512_mul_ph(a, b));
    }
    static __m512h combine(__m512h x, __m512h y) { return _mm512_add_ph(x, y); }
    static float reduce(__m512h x) {
        return static_cast<float>(_mm512_reduce_add_ph(x));
    }
};
#endif  // __AVX512FP16__

/////
///// Native AVX512-FP16 kernel for float (query) x Float16 (data) inner product.
/////
// IPMixedFloatToHalfOp32 converts the float32 query to fp16 on the fly and then
// performs the same vfmaddph computation as IPNativeHalfOp32.
// load_a handles the float query (needs down-conversion to fp16).
// load_b handles the Float16 data (loaded directly).

#if defined(__AVX512FP16__)
struct IPMixedFloatToHalfOp32 {
    static constexpr size_t simd_width = 32;
    using mask_t = svs::mask_repr_t<32>;  // uint32_t

    static __m512h init() { return _mm512_setzero_ph(); }

    // Query is float32 — load 32 floats (2×__m512), convert to __m512h
    static __m512h load_a(const float* p) {
        __m256i lo = _mm512_cvtps_ph(_mm512_loadu_ps(p),      _MM_FROUND_NO_EXC);
        __m256i hi = _mm512_cvtps_ph(_mm512_loadu_ps(p + 16), _MM_FROUND_NO_EXC);
        return _mm512_castsi512_ph(
            _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
    }
    static __m512h load_a(mask_t m, const float* p) {
        // For the ragged tail — load up to 32 floats masked.
        // Split the 32-bit mask into two 16-bit halves for the two __m512 loads.
        __mmask16 m_lo = (__mmask16)(m & 0xFFFF);
        __mmask16 m_hi = (__mmask16)(m >> 16);
        __m256i lo = _mm512_cvtps_ph(
            _mm512_maskz_loadu_ps(m_lo, p),      _MM_FROUND_NO_EXC);
        __m256i hi = _mm512_cvtps_ph(
            _mm512_maskz_loadu_ps(m_hi, p + 16), _MM_FROUND_NO_EXC);
        return _mm512_castsi512_ph(
            _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
    }

    // Data is Float16 — load directly
    static __m512h load_b(const Float16* p) {
        return _mm512_loadu_ph(reinterpret_cast<const _Float16*>(p));
    }
    static __m512h load_b(mask_t m, const Float16* p) {
        return _mm512_castsi512_ph(
            _mm512_maskz_loadu_epi16((__mmask32)m, reinterpret_cast<const void*>(p)));
    }

    static __m512h accumulate(__m512h acc, __m512h a, __m512h b) {
        return _mm512_fmadd_ph(a, b, acc);
    }
    static __m512h accumulate(mask_t m, __m512h acc, __m512h a, __m512h b) {
        return _mm512_mask_add_ph(acc, (__mmask32)m, acc, _mm512_mul_ph(a, b));
    }
    static __m512h combine(__m512h x, __m512h y) { return _mm512_add_ph(x, y); }
    static float reduce(__m512h x) {
        return static_cast<float>(_mm512_reduce_add_ph(x));
    }
};
#endif  // __AVX512FP16__

// Out-of-line definition of IPImpl<N, Float16, Float16, AVX512_FP16>::compute.
// The struct is declared in inner_product.h so every consumer TU resolves to this
// native kernel rather than the catch-all inheritance fallback.
// When __AVX512FP16__ is defined (gcc/clang with -mavx512fp16), we use the
// native vfmaddph instruction path. Otherwise we fall back to the AVX512
// fp32-accumulating path.
template <size_t N>
SVS_NOINLINE float IPImpl<N, Float16, Float16, AVX_AVAILABILITY::AVX512_FP16>::compute(
    const Float16* a, const Float16* b, lib::MaybeStatic<N> length
) {
#if defined(__AVX512FP16__)
    return svs::simd::generic_simd_op(IPNativeHalfOp32{}, a, b, length);
#else
    return IPImpl<N, Float16, Float16, AVX_AVAILABILITY::AVX512>::compute(a, b, length);
#endif
}

// Out-of-line definition of IPImpl<N, float, Float16, AVX512_FP16>::compute.
// Converts the float32 query to fp16 on the fly, then uses vfmaddph.
template <size_t N>
SVS_NOINLINE float IPImpl<N, float, Float16, AVX_AVAILABILITY::AVX512_FP16>::compute(
    const float* a, const Float16* b, lib::MaybeStatic<N> length
) {
#if defined(__AVX512FP16__)
    return svs::simd::generic_simd_op(IPMixedFloatToHalfOp32{}, a, b, length);
#else
    return IPImpl<N, float, Float16, AVX_AVAILABILITY::AVX512>::compute(a, b, length);
#endif
}

// Reuse the AVX512 instantiations but under a distinct availability tag so
// runtime selection can prefer FP16-capable code paths.
DISTANCE_L2_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_L2_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_L2_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_L2_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_L2_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_L2_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_L2_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_L2_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_L2_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX512_FP16);

DISTANCE_IP_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_IP_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_IP_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_IP_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_IP_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_IP_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_IP_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_IP_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_IP_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX512_FP16);

DISTANCE_CS_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_CS_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_CS_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_CS_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_CS_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_CS_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_CS_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_CS_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX512_FP16);
DISTANCE_CS_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX512_FP16);

} // namespace svs::distance

#endif
