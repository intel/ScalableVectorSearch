# Copyright (C) 2023 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit
# this software or the related documents without Intel's prior written
# permission.
#
# This software and the related documents are provided as is, with no
# express or implied warranties, other than those that are expressly stated
# in the License.

# Use this with CI to build for whatever version of python is currently configured in the
# runner.
import platform

def get_wheel_key():
    version = platform.python_version_tuple()
    key = "".join(version[0:-1])
    return f"cp{key}-manylinux_x86_64"

if __name__ == "__main__":
    # print(get_wheel_key(), end = "")
    print(get_wheel_key())
