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

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Scalable Vector Search'
copyright = '2023, Intel Corporation'
author = ''
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "breathe",              # Injest doxygen files.
    "sphinx.ext.autodoc",   # Generate documentation for Python interface.
    "sphinx.ext.napoleon",  # Allow simpler layouts for Python docstrings.
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.githubpages",
    "sphinx_collapse",  # Adds collapsible sections
    # "sphinx_autodoc_typehints",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Autodoc Configuration ---------------------------------------------------
autodoc_typehints = "description"
#autodoc_typehints_format = "short"

# -- Breathe Configuration ---------------------------------------------------
breathe_default_project = "SVS"
breathe_order_parameters_first = True

# The `breathe_projects` variable is passed via CMake as a command-line argument.
# breathe_projects = {}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    "css/custom.css",
]
