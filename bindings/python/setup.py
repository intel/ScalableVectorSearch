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

from skbuild import setup
import os

# If building in a cibuildwheel context, compile multiple versions of the library for
# optimized for various microarchitectures.
#
# This at least lets us have some kind of compatibility with older CPUs.
cmake_args = [
    # Export compile commands to allow us to explore compiler flags as needed.
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=YES",
    "-DSVS_EXPERIMENTAL_ENABLE_IVF=YES ",
    "-DSVS_EXPERIMENTAL_BUILD_CUSTOM_MKL=YES ",
]

# Determine the root of the repository
base_dir = os.path.relpath(os.path.join(os.path.dirname(__file__), '..', '..'))

project_urls = {
    "Source Code": "https://github.com/intel/ScalableVectorSearch",
    "Documentation": "https://intel.github.io/ScalableVectorSearch/",
    "Bug Tracker": "https://github.com/intel/ScalableVectorSearch",
}

with open(os.path.join(base_dir, "README.md"), "r", encoding="utf8") as f:
    long_description = f.read()

setup(
    name="scalable-vs",
    version="0.0.10",
    description="Scalable Vector Search (SVS) is a performance library for vector similarity search.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/intel/ScalableVectorSearch",
    author_email="onedal.maintainers@intel.com",
    maintainer_email="onedal.maintainers@intel.com",
    project_urls=project_urls,
    packages=['svs'],
    package_dir={'': 'src'},
    cmake_install_dir='src/svs',
    cmake_args = cmake_args,
    install_requires = [
        "numpy>=1.10.0, <3",   # keep in-sync with `pyproject.toml`
        "archspec>=0.2.0", # keep in-sync with `pyproject.toml`
        "toml>=0.10.2",    # keep in-sync with `pyproject.toml` required for the tests
    ],
    license="Apache License, Version 2.0",
    license_files=[
        os.path.join(base_dir, "LICENSE"),
        os.path.join(base_dir, "THIRD-PARTY-PROGRAMS"),
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Other Audience",
        "License :: OSI Approved :: Apache License, Version 2.0",
        "Topic :: Scientific/Engineering",
        "Topic :: System",
        "Topic :: Software Development",
    ],
    keywords=["machine learning", "data science", "data analytics"],
    include_package_data=True,
)
