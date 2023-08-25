# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import inspect
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath('../src'))
                
import sphinx
import numpy

# -- Project information -----------------------------------------------------

project = "DAGMA"
copyright = "2023, Kevin Bello"
author = "Kevin Bello"


# -- General configuration ---------------------------------------------------
# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_immaterial",
    "sphinx_immaterial.apidoc.python.apigen",
    # "myst_parser",
    "sphinx_design",
    'autoapi.extension',
]

autoapi_dirs = ['../src/dagma']

autosummary_generate = True
master_doc = "index"


intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

# templates_path = ["_templates"]

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_immaterial"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "extra.css",
]

html_title = "DAGMA"

html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
    },
    "site_url": "https://dagma.readthedocs.io/",
    "repo_url": "https://github.com/kevinsbello/dagma",
    "repo_name": "kevinsbello/dagma",
    # "repo_type": "github",
    "social": [
        {
            "icon": "fontawesome/brands/github",
            "link": "https://github.com/kevinsbello/dagma",
        },
        {
            "icon": "fontawesome/brands/python",
            "link": "https://pypi.org/project/dagma/",
        },
    ],
    "edit_uri": "",
    "globaltoc_collapse": False,
    "features": [
        # "navigation.expand",
        "navigation.tabs",
        # "toc.integrate",
        # "navigation.sections",
        # "navigation.instant",
        # "header.autohide",
        "navigation.top",
        "navigation.tracking",
        "toc.follow",
        "toc.sticky",
        "content.tabs.link",
        "announce.dismiss",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "toggle": {
                "icon": "material/weather-night",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "toggle": {
                "icon": "material/weather-sunny",
                "name": "Switch to light mode",
            },
        },
    ],
    "version_dropdown": False,
}

html_last_updated_fmt = ""
html_use_index = True
html_domain_indices = True


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

autodoc_default_options = {
    "imported-members": True,
    "members": True,
    # "special-members": True,
    # "inherited-members": "ndarray",
    # "member-order": "groupwise",
}
autodoc_typehints = "signature"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"

# myst_enable_extensions = ["dollarmath"]


# -- Sphinx Immaterial configs -------------------------------------------------

# Python apigen configuration
python_apigen_modules = {
      "dagma": "api/",
}

python_apigen_default_groups = [
    ("class:.*", "Classes"),
    ("data:.*", "Variables"),
    ("function:.*", "Functions"),
    ("classmethod:.*", "Class methods"),
    ("method:.*", "Methods"),
    (r"method:.*\.[A-Z][A-Za-z,_]*", "Constructors"),
    (r"method:.*\.__[A-Za-z,_]*__", "Special methods"),
    (r"method:.*\.__(init|new)__", "Constructors"),
    (r"method:.*\.__(str|repr)__", "String representation"),
    ("property:.*", "Properties"),
    (r".*:.*\.is_[a-z,_]*", "Attributes"),
]

python_apigen_default_order = [
    ("class:.*", 10),
    ("data:.*", 11),
    ("function:.*", 12),
    ("classmethod:.*", 40),
    ("method:.*", 50),
    (r"method:.*\.[A-Z][A-Za-z,_]*", 20),
    (r"method:.*\.__[A-Za-z,_]*__", 28),
    (r"method:.*\.__(init|new)__", 20),
    (r"method:.*\.__(str|repr)__", 30),
    ("property:.*", 60),
    (r".*:.*\.is_[a-z,_]*", 70),
]

python_apigen_order_tiebreaker = "alphabetical"
python_apigen_case_insensitive_filesystem = False
python_apigen_show_base_classes = True

# Python domain directive configuration
python_module_names_to_strip_from_xrefs = ["collections.abc"]

# General API configuration
object_description_options = [
    ("py:.*", dict(include_rubrics_in_toc=True)),
]

# sphinx_immaterial_custom_admonitions = [
#     {
#         "name": "seealso",
#         "title": "See also",
#         "classes": ["collapsible"],
#         "icon": "fontawesome/regular/eye",
#         "override": True,
#     },
#     {
#         "name": "star",
#         "icon": "octicons/star-16",
#         "color": (255, 233, 3),  # Gold
#     },
#     {
#         "name": "fast-performance",
#         "title": "Faster performance",
#         "icon": "material/speedometer",
#         "color": (40, 167, 69),  # Green: --sd-color-success
#     },
#     {
#         "name": "slow-performance",
#         "title": "Slower performance",
#         "icon": "material/speedometer-slow",
#         "color": (220, 53, 69),  # Red: --sd-color-danger
#     },
# ]


# -- Monkey-patching ---------------------------------------------------------

SPECIAL_MEMBERS = [
    "__repr__",
    "__str__",
    "__int__",
    "__call__",
    "__len__",
    "__eq__",
]


def autodoc_process_signature(app, what, name, obj, options, signature, return_annotation):
    signature = modify_type_hints(signature)
    return_annotation = modify_type_hints(return_annotation)
    return signature, return_annotation


def modify_type_hints(signature):
    """
    Fix shortening numpy type annotations in string annotations created with
    `from __future__ import annotations` that Sphinx can't process before Python
    3.10.

    See https://github.com/jbms/sphinx-immaterial/issues/161
    """
    if signature:
        signature = signature.replace("np", "~numpy")
    return signature


def monkey_patch_parse_see_also():
    """
    Use the NumPy docstring parsing of See Also sections for convenience. This automatically
    hyperlinks plaintext functions and methods.
    """
    # Add the special parsing method from NumpyDocstring
    method = sphinx.ext.napoleon.NumpyDocstring._parse_numpydoc_see_also_section
    sphinx.ext.napoleon.GoogleDocstring._parse_numpydoc_see_also_section = method

    def _parse_see_also_section(self, section: str):
        """Copied from NumpyDocstring._parse_see_also_section()."""
        lines = self._consume_to_next_section()

        # Added: strip whitespace from lines to satisfy _parse_numpydoc_see_also_section()
        for i in range(len(lines)):
            lines[i] = lines[i].strip()

        try:
            return self._parse_numpydoc_see_also_section(lines)
        except ValueError:
            return self._format_admonition("seealso", lines)

    sphinx.ext.napoleon.GoogleDocstring._parse_see_also_section = _parse_see_also_section
    
    
def autodoc_skip_member(app, what, name, obj, skip, options):
    """
    Instruct autodoc to skip members that are inherited from np.ndarray.
    """
    if skip:
        # Continue skipping things Sphinx already wants to skip
        return skip

    if name == "__init__":
        return False
    elif hasattr(obj, "__objclass__"):
        # This is a NumPy method, don't include docs
        return True
    elif getattr(obj, "__qualname__", None) in ["FunctionMixin.dot", "Array.astype"]:
        # NumPy methods that were overridden, don't include docs
        return True
    elif (
        hasattr(obj, "__qualname__")
        and getattr(obj, "__qualname__").split(".")[0] == "FieldArray"
        and hasattr(numpy.ndarray, name)
    ):
        if name in ["__repr__", "__str__"]:
            # Specifically allow these methods to be documented
            return False
        else:
            # This is a NumPy method that was overridden in one of our ndarray subclasses. Also don't include
            # these docs.
            return True

    if name in SPECIAL_MEMBERS:
        # Don't skip members in "special-members"
        return False

    if name[0] == "_":
        # For some reason we need to tell Sphinx to hide private members
        return True

    return skip


def autodoc_process_bases(app, name, obj, options, bases):
    """
    Remove private classes or mixin classes from documented class bases.
    """
    # Determine the bases to be removed
    remove_bases = []
    for base in bases:
        if base.__name__[0] == "_" or "Mixin" in base.__name__:
            remove_bases.append(base)

    # Remove from the bases list in-place
    for base in remove_bases:
        bases.remove(base)
        

def setup(app):
    monkey_patch_parse_see_also()
    app.connect("autodoc-skip-member", autodoc_skip_member)
    app.connect("autodoc-process-bases", autodoc_process_bases)
    app.connect("autodoc-process-signature", autodoc_process_signature)