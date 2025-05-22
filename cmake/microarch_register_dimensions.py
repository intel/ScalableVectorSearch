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
    parser.add_argument("--header-file", required=True)
    parser.add_argument("--dimensions", required=True, nargs="+", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    with open(args.header_file, "r") as f:
        header_content = f.read()

    prefix = "#define SVS_DISTANCE_TEMPLATES_BY_MICROARCH(dist, spec, uarch)"
    appendix = "\\\n    SVS_DISTANCE_FIXED_N_TEMPLATES_BY_MICROARCH(dist, spec, uarch, {dim})"
    for dim in args.dimensions:
        header_content = header_content.replace(prefix, prefix + appendix.format(dim=dim))

    with open(args.header_file, "w") as f:
        f.write(header_content)


if __name__ == "__main__":
    main()
