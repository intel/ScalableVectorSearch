# AGENTS.md — include/svs/quantization/

Quantization algorithms and data types (LVQ, scalar quantization).

- **Type safety:** Explicitly validate quantized dtype (uint8, int8, uint16) before operations.
- **Precision:** Changes to quantization schemes must preserve reconstruction error bounds. Document expected accuracy impact.
- **SIMD alignment:** Quantized data structures must respect SIMD lane widths (16/32/64 bytes for AVX-512).
- **Codec compatibility:** Do not break serialization format without versioning and migration path.

## Common failure modes
- Using `float` arithmetic in quantization hot paths (use integer SIMD intrinsics).
- Misaligned memory access in quantized vector loads/stores.
- Forgetting to update codebook when changing quantization levels.
