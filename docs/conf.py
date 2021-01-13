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

import os

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

extensions = [
    'sphinx_rtd_theme',
    'tensorstore_jsonschema_sphinx',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'tensorstore_autosummary',
    'sphinx.ext.mathjax',
]

exclude_patterns = [
    # This is included directly by `python/api/index.rst`, so we don't want to
    # generate a separate page for it.
    'python/api/tensorstore.rst',
    '_templates/**',
]

source_suffix = '.rst'
master_doc = 'index'
language = None

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']
templates_path = ['_templates']
html_context = {
    'css_files': ['_static/sphinx_rtd_theme_table_word_wrap_fix.css',],
}

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

mathjax_config = {
    'displayAlign': 'left',
}

tensorstore_jsonschema_id_map = {
    'https://github.com/google/tensorstore/json-schema/tensorstore':
        os.path.abspath('tensorstore_schema.yml'),
    'https://github.com/google/tensorstore/json-schema/dtype':
        os.path.abspath('tensorstore_schema.yml#/definitions/dtype'),
    'https://github.com/google/tensorstore/json-schema/driver/key-value-store-backed-chunk-driver':
        os.path.abspath('tensorstore/driver/kvs_backed_chunk_driver_schema.yml'
                       ),
    'https://github.com/google/tensorstore/json-schema/key-value-store':
        os.path.abspath('tensorstore/kvstore/schema.yml'),
}

always_document_param_types = True

autosummary_generate = True

doctest_global_setup = """
import tensorstore as ts
import numpy as np
"""
