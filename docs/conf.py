# Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

HERE = Path(__file__).parent


# -- Project information -----------------------------------------------------

# NOTE: If you installed your project in editable mode, this might be stale.
#       If this is the case, reinstall it to refresh the metadata
info = metadata("anndata-mcp")
project_name = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
repository_url = urls["Source"]

# The full version, including alpha/beta/rc tags
release = info["Version"]

# bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = True  # Warn about broken links
needs_sphinx = "4.0"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "dschaub95",
    "github_repo": project_name,
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_tabs.tabs",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    "autoapi.extension",
]
autoapi_dirs = ["../src/anndata_mcp"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "imported-members",
]
autoapi_keep_files = False
autoapi_add_toctree_entry = False  # We manually add to toctree in index.md
autoapi_python_use_implicit_namespaces = False
autoapi_prepend_jinja_directives = True
# Include private modules (files starting with _)
autoapi_file_patterns = ["*.py"]
# Configure AutoAPI to handle docstrings better
autoapi_type_aliases = {}
autoapi_ignore = []
# Mock imports that might cause issues during documentation generation
autoapi_mock_imports = []
autosummary_generate = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
myst_heading_anchors = 6  # create anchors for h1-h6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "click": ("https://click.palletsprojects.com/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Suppress warnings
# Note: Duplicate object descriptions from AutoAPI are expected when modules
# are documented both in the index and their own pages
suppress_warnings = ["app.add_directive", "ref.duplicate"]


def setup(app):
    """Configure Sphinx to suppress duplicate object description warnings from AutoAPI."""
    import logging

    logger = logging.getLogger("sphinx")

    def filter_duplicate_warnings(record):
        """Filter out duplicate object description warnings from AutoAPI."""
        if "duplicate object description" in str(record.msg):
            return False
        return True

    # Add filter to suppress duplicate object warnings
    logger.addFilter(filter_duplicate_warnings)
    return {"version": "0.1", "parallel_read_safe": True}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_title = project_name

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
}

pygments_style = "default"

nitpick_ignore = [
    # If building the documentation fails because of a missing link that is outside your control,
    # you can add an exception to this list.
    #     ("py:class", "igraph.Graph"),
    # External dependencies without intersphinx
    ("py:class", "fastmcp.FastMCP"),
    # Pydantic Field - handled by pydantic intersphinx but sometimes not found
    ("py:class", "Field"),
    ("py:obj", "pydantic.BaseModel"),
    # Type annotations that are not classes
    ("py:class", "optional"),
    # Internal types that may not be documented
    ("py:class", "Dataset2D"),
    ("py:class", "pd.DataFrame"),
    ("py:class", "pd.Index"),
    ("py:class", "dask.array.core.Array"),
    ("py:class", "dask.array.Array"),
    ("py:class", "pandas.DataFrame"),
    ("py:class", "pandas.Index"),
    # Internal classes in private modules (not documented by AutoAPI)
    ("py:class", "ExplorationResult"),
    ("py:class", "AnnDataSummary"),
    ("py:class", "DataView"),
]
