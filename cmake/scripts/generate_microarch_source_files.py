#!/usr/bin/env python3

import os
import sys
from itertools import product

# Base types
types = [
    "float const",
    "int8_t const",
    "uint8_t const",
    "svs::float16::Float16 const"
]

# Extra mixed type combinations
extra_type_pairs = [
    ("float const", "uint8_t const"),
    ("float const", "int8_t const"),
    ("float const", "svs::float16::Float16 const"),
    ("svs::float16::Float16 const", "float const"),
    ("svs::float16::Float16 const", "svs::float16::Float16 const")
]

# Unique (ea, eb) pairs only
type_pairs = set(product(types, repeat=2)).union(extra_type_pairs)

extents = [ "Dynamic", "64", "96", "100", "128", "512", "768" ]

arch_platform_map = {
    # "x86_64_v2": "defined(__x86_64__)",
    # "nehalem": "defined(__x86_64__)",
    # "westmere": "defined(__x86_64__)",
    # "sandybridge": "defined(__x86_64__)",
    # "ivybridge": "defined(__x86_64__)",
    "haswell": "defined(__x86_64__)",
    # "broadwell": "defined(__x86_64__)",
    # "skylake": "defined(__x86_64__)",
    "x86_64_v4": "defined(__x86_64__)",
    "skylake_avx512": "defined(__x86_64__)",
    "cascadelake": "defined(__x86_64__)",
    # "cooperlake": "defined(__x86_64__)",
    "icelake_client": "defined(__x86_64__)",
    # "icelake_server": "defined(__x86_64__)",
    "sapphirerapids": "defined(__x86_64__)",
    # "graniterapids": "defined(__x86_64__)",
    # "graniterapids_d": "defined(__x86_64__)",
    "m1": "defined(__aarch64__) && defined(__APPLE__)",
    "m2": "defined(__aarch64__) && defined(__APPLE__)",
    "neoverse_v1": "defined(__aarch64__) && !defined(__APPLE__)",
    "neoverse_n2": "defined(__aarch64__) && !defined(__APPLE__)",
}


HEADER = '''#include <span>

#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/inner_product.h"
#include "svs/core/distance/cosine.h"
#include "svs/core/distance/distance_core.h"

// clang-format off'''

FOOTER = '''// clang-format on
#endif // ARCH GUARD
'''


def write_cpp_file(arch: str, output_dir: str):
    filename = f"microarch_{arch}.cpp"
    path = os.path.join(output_dir, filename)

    guard = arch_platform_map[arch]
    arch_enum = f"svs::arch::MicroArch::{arch}"

    lines = [HEADER]
    lines.append(f"#if {guard}")
    lines.append("namespace svs::distance {\n")

    for ea, eb in sorted(type_pairs):
        for extent in extents:
            lines.append(f"template float compute<{ea}, {eb}, {extent}, {extent}, {arch_enum}>(DistanceL2<{arch_enum}>, std::span<{ea}, {extent}>, std::span<{eb}, {extent}>);")
            lines.append(f"template float compute<{ea}, {eb}, {extent}, {extent}, {arch_enum}>(DistanceIP<{arch_enum}>, std::span<{ea}, {extent}>, std::span<{eb}, {extent}>);")
            lines.append(f"template float compute<{ea}, {eb}, {extent}, {extent}, {arch_enum}>(DistanceCosineSimilarity<{arch_enum}>, std::span<{ea}, {extent}>, std::span<{eb}, {extent}>);")

    lines.append("} // namespace svs::distance")
    lines.append(FOOTER)

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated: {path}")


def write_header_file(output_dir: str):
    path = os.path.join(output_dir, "distance_instantiations.h")
    lines = [
        "#pragma once",
        "#include <span>",
        "#include \"svs/core/distance/euclidean.h\"",
        "#include \"svs/core/distance/inner_product.h\"",
        "#include \"svs/core/distance/cosine.h\"",
        "#include \"svs/core/distance/distance_core.h\"",
        "namespace svs::distance {"
    ]

    for arch, guard in arch_platform_map.items():
        arch_enum = f"svs::arch::MicroArch::{arch}"
        lines.append(f"#if {guard}")
        for ea, eb in product(types, repeat=2):
            for extent in extents:
                lines.append(f"extern template float compute<{ea}, {eb}, {extent}, {extent}, {arch_enum}>(DistanceL2<{arch_enum}>, std::span<{ea}, {extent}>, std::span<{eb}, {extent}>);")
                lines.append(f"extern template float compute<{ea}, {eb}, {extent}, {extent}, {arch_enum}>(DistanceIP<{arch_enum}>, std::span<{ea}, {extent}>, std::span<{eb}, {extent}>);")
                lines.append(f"extern template float compute<{ea}, {eb}, {extent}, {extent}, {arch_enum}>(DistanceCosineSimilarity<{arch_enum}>, std::span<{ea}, {extent}>, std::span<{eb}, {extent}>);")
        lines.append(f"#endif // {guard}")
    lines.append("} // namespace svs::distance")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated: {path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: generate_distance_instantiations.py <output_directory>", file=sys.stderr)
        sys.exit(1)

    output_dir = sys.argv[1]
    os.makedirs(output_dir, exist_ok=True)

    for arch in arch_platform_map:
        write_cpp_file(arch, output_dir)

    write_header_file(output_dir)


if __name__ == "__main__":
    main()
