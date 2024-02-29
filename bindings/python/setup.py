from skbuild import setup
import archspec.cpu as cpu
import os

# If building in a cibuildwheel context, compile multiple versions of the library for
# optimized for various microarchitectures.
#
# This at least lets us have some kind of compatibility with older CPUs.
cmake_args = [
    # Export compile commands to allow us to explore compiler flags as needed.
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=YES",
    # TODO: need to remove these LEANVEC and MKL dependency from here
    "-DSVS_EXPERIMENTAL_LEANVEC=YES ",
    "-DSVS_EXPERIMENTAL_BUILD_CUSTOM_MKL=YES ",
]

# Utility to convert micro-architecture strings to
def target(arch):
    return cpu.TARGETS[arch]

# N.B.: cibuildwheel must configure the multi-arch environment variable.
# Also, the micro-architectures defined below should be in order of preference.
if os.environ.get("PYSVS_MULTIARCH", None) is not None:
    pysvs_microarchs = [
        "cascadelake",
        "x86_64_v3", # conservative base CPU for x86 CPUs.
    ]

    # Add the current host to the list of micro-architecture if it doesn't already exist.
    last_target = target(pysvs_microarchs[-1])
    host_name = cpu.host().name
    if host_name not in pysvs_microarchs and target(host_name) < last_target:
        pysvs_microarchs.append(host_name)

    cmake_array = ";".join(pysvs_microarchs)
    cmake_args.append(f"-DPYSVS_MICROARCHS={cmake_array}")

setup(
    name="pysvs",
    version="0.0.4",
    packages=['pysvs'],
    package_dir={'': 'src'},
    cmake_install_dir='src/pysvs',
    cmake_args = cmake_args,
    install_requires = [
        "numpy>=1.10.0",   # keep in-sync with `pyproject.toml`
        "archspec>=0.2.0", # keep in-sync with `pyproject.toml`
        "toml>=0.10.2",    # keep in-sync with `pyproject.toml` required for the tests
    ]
)
