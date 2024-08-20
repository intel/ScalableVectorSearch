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
    # TODO: need to remove these LEANVEC and Intel(R) MKL dependency from here
    "-DSVS_EXPERIMENTAL_LEANVEC=YES ",
    "-DSVS_EXPERIMENTAL_BUILD_CUSTOM_MKL=YES ",
]

# Utility to convert micro-architecture strings to
def target(arch):
    return cpu.TARGETS[arch]

# N.B.: cibuildwheel must configure the multi-arch environment variable.
# Also, the micro-architectures defined below should be in order of preference.
if os.environ.get("SVS_MULTIARCH", None) is not None:
    svs_microarchs = [
        "cascadelake",
        "x86_64_v3", # conservative base CPU for x86 CPUs.
    ]

    # Add the current host to the list of micro-architecture if it doesn't already exist.
    last_target = target(svs_microarchs[-1])
    host_name = cpu.host().name
    if host_name not in svs_microarchs and target(host_name) < last_target:
        svs_microarchs.append(host_name)

    cmake_array = ";".join(svs_microarchs)
    cmake_args.append(f"-DSVS_MICROARCHS={cmake_array}")

# Determine the root of the repository
base_dir = os.path.relpath(os.path.join(os.path.dirname(__file__), '..', '..'))

setup(
    name="scalable-vs",
    version="0.0.4",
    packages=['svs'],
    package_dir={'': 'src'},
    cmake_install_dir='src/svs',
    cmake_args = cmake_args,
    install_requires = [
        "numpy>=1.10.0, <2",   # keep in-sync with `pyproject.toml`
        "archspec>=0.2.0", # keep in-sync with `pyproject.toml`
        "toml>=0.10.2",    # keep in-sync with `pyproject.toml` required for the tests
    ],
    license="GNU Affero General Public License v3 or later (AGPLv3+)",
    license_files=[
        os.path.join(base_dir, "LICENSE"),
        os.path.join(base_dir, "THIRD-PARTY-PROGRAMS"),
    ],
    classifiers=[
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    ],
    include_package_data=True,
)
