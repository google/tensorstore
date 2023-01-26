# Copyright 2020 The TensorStore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sphinx configuration for TensorStore."""

from typing import Optional, NamedTuple

import docutils.nodes
import sphinx.addnodes
import sphinx.domains.python
import sphinx.environment

project = 'TensorStore'
copyright = '2020 The TensorStore Authors'  # pylint: disable=redefined-builtin

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = ''

# Override default of `utf-8-sig` which can cause problems with autosummary due
# to the extra Unicode Byte Order Mark that gets inserted.
source_encoding = 'utf-8'

# Don't include "View page source" links, since they aren't very helpful,
# especially for generated pages.
html_show_sourcelink = False
html_copy_source = False

# Skip unnecessary footer text.
html_show_sphinx = False
html_show_copyright = False

extensions = [
    'sphinx.ext.extlinks',
    'sphinx_immaterial',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'tensorstore_sphinx_ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx_immaterial.apidoc.format_signatures',
    'sphinx_immaterial.apidoc.cpp.cppreference',
    'sphinx_immaterial.apidoc.json.domain',
    'sphinx_immaterial.apidoc.python.apigen',
]

exclude_patterns = [
    # Included by installation.rst
    'third_party_libraries.rst',
    '_templates/**',
]

source_suffix = '.rst'
master_doc = 'index'
language = 'en'

html_theme = 'sphinx_immaterial'

html_title = 'TensorStore'

# html_logo = 'logo.svg'

html_use_index = False

html_favicon = '_templates/logo.svg'

html_theme_options = {
    'logo_svg':
        'logo.svg',
    'site_url':
        'https://google.github.io/tensorstore/',
    'repo_url':
        'https://github.com/google/tensorstore/',
    'repo_name':
        'google/tensorstore',
    'repo_type':
        'github',
    'globaltoc_collapse':
        True,
    'features': [
        'navigation.expand',
        # 'navigation.tabs',
        # 'toc.integrate',
        'navigation.sections',
        # 'navigation.instant',
        # 'header.autohide',
        'navigation.top',
        # 'search.highlight',
        # 'search.share',
        "toc.follow",
        "toc.sticky",
    ],
    'toc_title_is_page_title':
        True,
    'palette': [
        {
            'media': '(prefers-color-scheme: dark)',
            'scheme': 'slate',
            'primary': 'green',
            'accent': 'light blue',
            'toggle': {
                'icon': 'material/lightbulb',
                'name': 'Switch to light mode',
            },
        },
        {
            'media': '(prefers-color-scheme: light)',
            'scheme': 'default',
            'primary': 'green',
            'accent': 'light blue',
            'toggle': {
                'icon': 'material/lightbulb-outline',
                'name': 'Switch to dark mode',
            },
        },
    ],
}

templates_path = ['_templates']

intersphinx_mapping = {
    'python':
        ('https://docs.python.org/3', ('intersphinx_inv/python3.inv', None)),
    'zarr': ('https://zarr.readthedocs.io/en/stable',
             ('intersphinx_inv/zarr.inv', None)),
    'numpy':
        ('https://numpy.org/doc/stable/', ('intersphinx_inv/numpy.inv', None)),
    'dask': ('https://docs.dask.org/en/latest/', ('intersphinx_inv/dask.inv',
                                                  None)),
}

rst_prolog = """
.. role:: python(code)
   :language: python
   :class: highlight

.. role:: json(code)
   :language: json
   :class: highlight
"""

# Warn about missing references
nitpicky = True

default_role = 'any'

# Extension options
# -----------------

# Use MathJax 3.
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

mathjax3_config = {
    'chtml': {
        'displayAlign': 'left',
    },
}

always_document_param_types = True

doctest_global_setup = """
import tensorstore as ts
import numpy as np
"""

extlinks = {
    'wikipedia': ('https://en.wikipedia.org/wiki/%s', None),
}

napoleon_numpy_docstring = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True

json_schemas = [
    "*schema.yml",
    "**/*schema.yml",
]

json_schema_rst_prolog = """
.. default-role:: json:schema

.. default-literal-role:: json

.. highlight:: json
"""

python_apigen_modules = {"tensorstore": "python/api/tensorstore."}

python_apigen_default_groups = [
    ("class:.*", "Classes"),
    (r".*:.*\.__(init|new)__", "Constructors"),
    (r".*:.*\.__eq__", "Comparison operators"),
    (r".*:.*\.__(str|repr)__", "String representation"),
]

python_apigen_rst_prolog = """
.. default-role:: py:obj

.. default-literal-role:: python

.. highlight:: python

"""

python_module_names_to_strip_from_xrefs = ["tensorstore"]

python_type_aliases = {
    "dtype": "numpy.dtype",
    "Real": "numbers.Real",
}

python_strip_property_prefix = True


# Monkey patch Sphinx to generate custom cross references for specific type
# annotations.
#
# The Sphinx Python domain generates a `py:class` cross reference for type
# annotations.  However, in some cases in the TensorStore documentation, type
# annotations are used to refer to targets that are not actual Python classes,
# such as `DownsampleMethod`, `DimSelectionLike`, or `NumpyIndexingSpec`.
# Additionally, some types like `numpy.typing.ArrayLike` are `py:data` objects
# and can't be referenced as `py:class`.
class TypeXrefTarget(NamedTuple):
  domain: str
  reftype: str
  target: str
  title: str


python_type_to_xref_mappings = {
    "numpy.typing.ArrayLike":
        TypeXrefTarget("py", "data", "numpy.typing.ArrayLike", "ArrayLike"),
    "NumpyIndexingSpec":
        TypeXrefTarget("std", "ref", "python-numpy-style-indexing",
                       "NumpyIndexingSpec"),
    "DimSelectionLike":
        TypeXrefTarget("std", "ref", "python-dim-selections",
                       "DimSelectionLike"),
    "DownsampleMethod":
        TypeXrefTarget("json", "schema", "DownsampleMethod",
                       "DownsampleMethod"),
}

_orig_python_type_to_xref = sphinx.domains.python.type_to_xref


def _python_type_to_xref(
    target: str, env: Optional[sphinx.environment.BuildEnvironment] = None,
    suppress_prefix: bool = False) -> sphinx.addnodes.pending_xref:
  xref_info = python_type_to_xref_mappings.get(target)
  if xref_info is not None:
    return sphinx.addnodes.pending_xref(
        '',
        docutils.nodes.Text(xref_info.title),
        refdomain=xref_info.domain,
        reftype=xref_info.reftype,
        reftarget=xref_info.target,
        refspecific=False,
        refexplicit=True,
        refwarn=True,
    )
  return _orig_python_type_to_xref(target, env, suppress_prefix)


sphinx.domains.python.type_to_xref = _python_type_to_xref


def setup(app):

  # Exclude pybind11-builtin base class when displaying base classes.
  #
  # This base class is purely an implementation detail and not helpful to
  # display to users.
  def _autodoc_process_bases(app, name, obj, options, bases):
    bases[:] = [
        base for base in bases if base.__module__ != "pybind11_builtins"
    ]

  app.connect("autodoc-process-bases", _autodoc_process_bases)
