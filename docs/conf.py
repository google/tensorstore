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
    'tensorstore_sphinx_material.sphinx_material',
    'tensorstore_sphinx_ext.jsonschema_sphinx',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'tensorstore_sphinx_ext.autodoc',
    'tensorstore_sphinx_ext.autosummary',
    'tensorstore_sphinx_ext.doctest',
    'sphinx.ext.mathjax',
]

exclude_patterns = [
    # Included by installation.rst
    'third_party_libraries.rst',
    '_templates/**',
]

source_suffix = '.rst'
master_doc = 'index'
language = 'en'

html_theme = 'sphinx_material'

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
    'globaltoc_depth':
        -1,
    'globaltoc_collapse':
        True,
    'globaltoc_includehidden':
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
    ],
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
