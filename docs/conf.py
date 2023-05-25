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
