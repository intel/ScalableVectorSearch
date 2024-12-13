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

# dep pre-coms
import archspec.cpu as cpu

# standard library
import json
import importlib
import os
from pathlib import Path

# Get environment variables for configuring warnings and overriding backend selection.
def _is_quiet():
    """
    Return whether or not backend loading should be "quiet".
    In this context, "quiet" means not warning for older architectures.
    """
    return os.environ.get("SVS_QUIET", False)

def _override_backend():
    """
    Return a manual override for the backend.
    If no override is set, return `None`.
    """
    return os.environ.get("SVS_OVERRIDE_BACKEND", None)


# The name of the manifest file.
FLAGS_MANIFEST = "flags_manifest.json" # Keep in-sync with CMakeLists.txt

def _library_from_suffix(suffix):
    return f"._svs_{suffix}"

def _message_prehook(spec, host = cpu.host()):
    """
    Emit any special messages for the given microarchitecture spec.
    """
    if _is_quiet():
        return

    if isinstance(spec, str):
        spec = cpu.TARGETS[spec]

    import warnings
    if spec <= cpu.TARGETS["skylake_avx512"]:
        message = f"""
        Loading library for an older CPU architecture ({spec}).
        Performance may be degraded.
        """
        warnings.warn(message, RuntimeWarning)

    if host < spec:
        message = """
        Override backend is target for a newer CPU than the one you're currently using.
        Application may crash.
        """
        warnings.warn(message, RuntimeWarning)


# The backend being used for this session
__CURRENT_BACKEND__ = None
def current_backend():
    """
    Return the name of the current backend.
    """
    return __CURRENT_BACKEND__

def __set_backend_once__(suffix: str, spec):
    global __CURRENT_BACKEND__
    if __CURRENT_BACKEND__ == None:
        _message_prehook(spec)
        __CURRENT_BACKEND__ = str(suffix)

    return current_backend()

# The dynamically loaded module.
__LIBRARY__ = None

def _load_manifest():
    """
    Determine which shared library to load to supply the C++ extentions.
    """
    json_file = Path(__file__).parent / FLAGS_MANIFEST
    json_file_alternate = Path(__file__).parent.parent / FLAGS_MANIFEST

    # Try to give a somewhat helpful error message if the JSON manifest file was not
    # generated properly by Scikit-build/CMake
    if json_file.exists():
        with open(json_file, "r") as io:
            return json.load(io)
    elif json_file_alternate.exists():
        with open(json_file_alternate, "r") as io:
            return json.load(io)
    else:
        print(Path(str(json_file).replace("ai.similarity-search.gss/", "")))
        raise RuntimeError(f"""
        Expected a file {FLAGS_MANIFEST} to exist in the source directory to describe the
        attributes of the libraries bundled with this application.

        No such file was found.

        Please report this to the project maintainer!
        """)

def available_backends():
    """
    Return a list of the available backends that where compiled when this module was built.

    Each backend in the list may be used to initialize ``SVS_OVERRIDE_BACKEND``
    environment variable prior to application start to override the default loading logic.
    """
    return list(_load_manifest()["libraries"].keys())

def _find_library():
    """
    Find the appropriate library to load for this micro architecture.
    """

    # Get the current CPU and the manifest of compiled libraries that ship with this
    # library.
    host = cpu.host()
    manifest = _load_manifest()

    # Respect override requests.
    # Down stream loading will fail if the given option doesn't exist.
    #
    # However, if an override is explicitly given, then we can assume that the use knows
    # what they're doing and can respond to a loading failure correctly.
    override = _override_backend()
    if override is not None:
        spec = cpu.TARGETS[manifest["libraries"][override]]
        return __set_backend_once__(override, spec)

    # Assume architectures in the manifest are place in order of preference.
    # TODO: Revisit this assumption.
    for (suffix, microarch) in manifest["libraries"].items():
        # Are we compatible with this micro architecture?
        spec = cpu.TARGETS[microarch]
        if spec <= host:
            return __set_backend_once__(suffix, spec)

    raise RuntimeError(f"""
        Could not find a suitable backend for your machine ({host}).
        Please contact the project maintainers!
        """)

def __load_module_once__():
    global __LIBRARY__
    if __LIBRARY__ is None:
        library_name = _library_from_suffix(_find_library())
        __LIBRARY__ = importlib.import_module(library_name, package = "svs")

def library():
    """
    Return the library backend as a module. Dynamically loads the library when first called.

    Dynamically loading the library may trigger warnings related to correctness or
    performance. If you really **really** don't want these warnings, they can be suppressed
    by defining the environemtn variable ``SVS_QUIET=YES`` prior to application start.
    """
    __load_module_once__()
    return __LIBRARY__
