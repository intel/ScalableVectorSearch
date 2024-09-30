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

# Test the dynamic loading logic
import archspec.cpu as cpu
import unittest
import os
import warnings

import svs.loader as loader

def set_quiet():
    os.environ["SVS_QUIET"] = "YES"

def clear_quiet():
    os.environ.pop("SVS_QUIET", None)

def set_override(override: str):
    os.environ["SVS_OVERRIDE_BACKEND"] = override

def clear_override():
    os.environ.pop("SVS_OVERRIDE_BACKEND", None)

class LoadingTester(unittest.TestCase):
    def __unset_environment_variables__(self):
        clear_quiet()
        clear_override()

    def tearDown(self):
        self.__unset_environment_variables__()

    def test_environment_variables(self):
        # Clear the environment variables in question.
        self.__unset_environment_variables__()

        # Make sure "is_quiet" behaves correctly.
        self.assertFalse(loader._is_quiet())
        set_quiet()
        self.assertTrue(loader._is_quiet())
        self.__unset_environment_variables__()
        self.assertFalse(loader._is_quiet())

        # Now, check that "override_backend" works.
        self.assertEqual(loader._override_backend(), None)
        set_override("hello")
        self.assertEqual(loader._override_backend(), "hello")
        set_override("north")
        self.assertEqual(loader._override_backend(), "north")
        clear_override()
        self.assertEqual(loader._override_backend(), None)
        self.__unset_environment_variables__()

    def test_suffix(self):
        self.assertEqual(loader._library_from_suffix("native"), "._svs_native")
        self.assertEqual(loader._library_from_suffix("cascadelake"), "._svs_cascadelake")

    def test_available_backends(self):
        self.assertGreaterEqual(len(loader.available_backends()), 1)

    def test_manifest(self):
        manifest = loader._load_manifest()
        self.assertTrue("toolchain" in manifest)
        self.assertTrue("libraries" in manifest)

        toolchain = manifest["toolchain"]
        self.assertTrue("compiler" in toolchain)
        self.assertTrue("compiler_version" in toolchain)

        libraries = manifest["libraries"]
        self.assertGreaterEqual(len(libraries), 1)

    def test_message_prehook(self):
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        # Refer to
        # https://docs.python.org/3/library/warnings.html#testing-warnings
        # for how to test warnings.

        # Warning for the host being greater than the spec.
        spec = cpu.TARGETS["icelake"]
        host = cpu.TARGETS["skylake"]
        with warnings.catch_warnings(record = True) as w:
            loader._message_prehook(spec, host)
            self.assertTrue(len(w) == 1)
            self.assertTrue(issubclass(w[-1].category, RuntimeWarning))
            self.assertTrue("Override" in str(w[-1].message))

        # Running again with "quiet" enabled should suppress the warning
        set_quiet()
        with warnings.catch_warnings(record = True) as w:
            loader._message_prehook(spec, host)
            self.assertTrue(len(w) == 0)

        # Warning for using an old architecture.
        clear_quiet()
        archs = ["haswell", "skylake", "skylake_avx512"]
        for arch in archs:
            with warnings.catch_warnings(record = True) as w:
                loader._message_prehook(arch)
                # Number of warnings can exceed 1 if running on an older CPU.
                # In this latter case, we get a "newer CPU" warning as well.
                self.assertTrue(len(w) >= 1)
                self.assertTrue(issubclass(w[0].category, RuntimeWarning))
                self.assertTrue("older CPU" in str(w[0].message))

    def test_loaded(self):
        libraries = loader._load_manifest()["libraries"]
        self.assertTrue(loader.current_backend() in libraries)
        self.assertNotEqual(loader.library(), None)
