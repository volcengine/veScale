# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pytorch_sphinx_theme
import torchdistx

# -- Project Information -----------------------------------------------------

project = "torchdistX"

copyright = "Meta Platforms, Inc. and affiliates"

author = "Pytorch Distributed Team"

version = torchdistx.__version__
release = torchdistx.__version__

# -- General Configuration ---------------------------------------------------

needs_sphinx = "4.3.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

autodoc_typehints = "description"
autodoc_typehints_format = "short"

todo_include_todos = True

intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- Options for HTML Output -------------------------------------------------

html_theme = "pytorch_sphinx_theme"

html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

html_theme_options = {
    "analytics_id": "UA-117752657-2",
    "collapse_navigation": False,
    "logo_only": True,
    "pytorch_project": "torchdistx",
}
