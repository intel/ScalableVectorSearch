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


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto-header-file", required=True)
    parser.add_argument("--output-header-file", required=True)
    parser.add_argument("--known-uarchs", required=True, nargs="+", type=str)
    parser.add_argument("--supported-uarchs", required=True, nargs="+", type=str)
    return parser


def main():
    args = build_parser().parse_args()
    with open(args.proto_header_file, "r") as f:
        header_content = f.read()

    # generate SVS_FOR_EACH_MICROARCH macro
    prefix = "#define SVS_FOR_EACH_MICROARCH"
    appendix = "\\\n    SVS_MICROARCH_FUNC({uarch})"
    for uarch in args.supported_uarchs[::-1]:
        header_content = header_content.replace(prefix, prefix + appendix.format(uarch=uarch))
    # generate SVS_FOR_EACH_KNOWN_MICROARCH macro
    prefix = "#define SVS_FOR_EACH_KNOWN_MICROARCH"
    appendix = "\\\n    SVS_MICROARCH_FUNC({uarch})"
    for uarch in args.known_uarchs[::-1]:
        header_content = header_content.replace(prefix, prefix + appendix.format(uarch=uarch))

    with open(args.output_header_file, "w") as f:
        f.write(header_content)


if __name__ == "__main__":
    main()
