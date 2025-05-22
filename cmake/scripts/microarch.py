# Copyright 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# A helper file that takes a list of CPU microarchitectures and generates:
#
# (1) A text file with compiler optimization flags for each microarchitecture formatted for
#     relatively easy consumption by CMake.
#
import archspec.cpu as cpu
import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cmake_flags_text_file",
        help = "file path to where CMake's text file will go."
    )
    parser.add_argument("--compiler", required = True)
    parser.add_argument("--compiler-version", required = True)
    parser.add_argument(
        "--microarchitectures",
        required = True,
        nargs = "+"
    )

    return parser

def resolve_microarch(name: str):
    """
    Allow for custom micro-architecture names.
    """
    custom_aliases = {
        "native": cpu.host().name,
        "icelake_client": "icelake",
    }
    # Allow the custom aliases to override the current name.
    # If an alias doesn't exist, juse pass the name straight through.
    return custom_aliases.get(name, name)

def dump_flags_for_cmake(flags: list, path: str):
    """
    Save the optimization flags to a text file suitable for CMake to ingest easily.

    Each entry in `flags` will be interpreted as a set of compiler flags for some
    microarchitecture. By default, archspec passes this as a space-delimited string.
    Because CMake has this tendency to treat both spaces and semi-colons as array
    separators, we introduce a comma between each flag set to assist in post-procfessing
    on the CMake side.

    We generate one such super-string per entry in `flags` and store the file at the
    file pointed to by `path`.

    Args:
        flags - A list of optimization flags.
        path - The file path where the output text file will be generated.
    """
    # white-space separated to comma-separated & one architecture per line
    string = "\n".join([",".join(f.split()) for f in flags])
    with open(path, "w") as file:
        file.write(string)

def resolve_compiler(name: str):
    """
    Convert compiler names from CMake land to archspec land.
    """
    aliases = {
        "GNU": "gcc",
        "Clang": "clang",
        "AppleClang": "clang",
        "IntelLLVM": "oneapi",
    }
    return aliases.get(name, name)

def run():
    parser = build_parser()
    args = parser.parse_args()

    # Extract elements from the parser
    architectures = args.microarchitectures
    output_text = args.cmake_flags_text_file
    compiler = resolve_compiler(args.compiler)
    compiler_version = args.compiler_version

    # Generate optimization flags.
    suffix_to_microarch = {}
    optimization_flags = []

    for arch in architectures:
        resolved = resolve_microarch(arch)
        suffix_to_microarch[arch] = resolved
        flags = cpu.TARGETS[resolved].optimization_flags(compiler, compiler_version)
        optimization_flags.append(flags)

    # Safe flags to file
    dump_flags_for_cmake(optimization_flags, output_text)

#####
##### Execute as script.
#####

if __name__ == "__main__":
    run()
