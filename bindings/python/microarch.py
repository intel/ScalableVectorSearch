# A helper file that takes a list of CPU microarchitectures and generates:
#
# (1) A text file with compiler optimization flags for each microarchitecture formatted for
#     relatively easy consumption by CMake.
#
# (2) A JSON manifest file describing the micreoarchitecture for each compiled library
#     that the python library can use to select the correct shared library.
#
import archspec
import archspec.cpu as cpu
import argparse
import json

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cmake_flags_text_file",
        help = "file path to where CMake's text file will go."
    )
    parser.add_argument("python_output_json_file")
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
    }
    # Allow the custom aliases to override the current name.
    # If an alias doesn't exist, juse pass the name straight through.
    return custom_aliases.get(name, name)

def dump_flags_for_cmake(flags: list, path: str):
    """
    Save the optimization flags to a text file suitable for CMake to injest easily.

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
    with open(path, "w") as file:
        num_flags = len(flags)
        for i, flag_set in enumerate(flags):
            file.write(",".join(flag_set.split()))
            # Add a new line if not the last flag set.
            if i != (num_flags - 1):
                file.write('\n')

def resolve_compiler(name: str):
    """
    Convert compiler names from CMake land to archspec land.
    """
    aliases = {
        "GNU": "gcc",
        "Clang": "clang",
    }
    return aliases.get(name, name)

def run():
    parser = build_parser()
    args = parser.parse_args()

    # Extract elements from the parser
    architectures = args.microarchitectures
    output_text = args.cmake_flags_text_file
    output_json = args.python_output_json_file
    compiler = resolve_compiler(args.compiler)
    compiler_version = args.compiler_version

    # Communicate the compiler environment to the python runtime.
    toolchain = {
        "compiler": compiler,
        "compiler_version": compiler_version,
    }
    suffix_to_microarch = {}
    optimization_flags = []

    # Generate optimization flags.
    for arch in architectures:
        resolved = resolve_microarch(arch)
        suffix_to_microarch[arch] = resolved
        flags = cpu.TARGETS[resolved].optimization_flags(compiler, compiler_version)
        optimization_flags.append(flags)

    # Dump the JSON output
    pre_json_dict = {
        "toolchain": toolchain,
        "libraries": suffix_to_microarch,
    }
    with open(output_json, "w") as file:
        file.write(json.dumps(pre_json_dict, indent = 4))

    # Safe flags to file
    dump_flags_for_cmake(optimization_flags, output_text)

    # Print flags to stdout
    for flags in optimization_flags:
        print(flags)

#####
##### Execute as script.
#####

if __name__ == "__main__":
    run()
