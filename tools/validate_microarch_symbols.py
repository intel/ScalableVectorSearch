# Copyright 2025 Intel Corporation
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

import argparse
import os
import subprocess
import sys


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validation script to check all microarch-specific " \
        "symbols in base object files are undefined."
    )
    parser.add_argument(
        "directories", nargs="+",
        help="Directories to recursively search for .cpp.o files"
    )
    return parser


def find_cpp_o_files(directory):
    cpp_o_files = []

    for root, dirs, files in os.walk(directory):
        # Skip directories with microarch-specific object-files
        dirs[:] = [d for d in dirs if 'microarch' not in d.lower()]

        for file in files:
            if file.endswith('.cpp.o'):
                cpp_o_files.append(os.path.join(root, file))

    return cpp_o_files


def get_microarch_symbols(obj_file):
    """Extract all defined microarch-specific symbols."""
    try:
        # Use nm to get the symbols from the object file
        result = subprocess.run(["nm", "-C", obj_file], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error processing {obj_file}: {result.stderr}")
            return []

        lines = result.stdout.splitlines()
        microarch_symbols = []

        for line in lines:
            if len(line.split()) > 2:
                _, symbol_type, symbol = line.split(maxsplit=2)
                if ("<(svs::arch::MicroArch" in symbol and
                symbol_type in ('T', 't', 'W', 'w') and
                "<(svs::arch::MicroArch)0" not in symbol):
                    microarch_symbols.append((obj_file, symbol))

        return microarch_symbols

    except Exception as e:
        print(f"Error processing {obj_file}: {str(e)}")
        return []


def main():
    args = build_parser().parse_args()

    cpp_o_files = []
    for directory in args.directories:
        dir_files = find_cpp_o_files(directory)
        print(f"Found {len(dir_files)} .cpp.o files in {directory}")
        cpp_o_files.extend(dir_files)

    print(f"Found a total of {len(cpp_o_files)} .cpp.o files in all directories")

    if not cpp_o_files:
        print("No .cpp.o files found in any of the provided directories")
        return

    all_symbols = []
    for obj_file in cpp_o_files:
        symbols = get_microarch_symbols(obj_file)
        all_symbols.extend(symbols)

    if all_symbols:
        print(f"Found {len(all_symbols)} defined MicroArch symbols:")
        for obj_file, symbol in all_symbols:
            print(f"{obj_file}: {symbol}")
        sys.exit(1)
    else:
        print("No defined MicroArch symbols found")
        sys.exit(0)


if __name__ == "__main__":
    main()
