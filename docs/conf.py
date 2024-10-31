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

import importlib
import os
import sys
from typing import NamedTuple, Optional

import docutils.nodes
import sphinx.addnodes
import sphinx.domains.python
import sphinx.environment
import sphinx.util.parallel

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
    'sphinx_immaterial.apidoc.cpp.apigen',
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
    'logo_svg': 'logo.svg',
    'site_url': 'https://google.github.io/tensorstore/',
    'repo_url': 'https://github.com/google/tensorstore/',
    'globaltoc_collapse': True,
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
        'toc.follow',
        'toc.sticky',
    ],
    'toc_title_is_page_title': True,
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
    'python': (
        'https://docs.python.org/3',
        ('intersphinx_inv/python3.inv', None),
    ),
    'zarr': (
        'https://zarr.readthedocs.io/en/stable',
        ('intersphinx_inv/zarr.inv', None),
    ),
    'zarr-specs': (
        'https://zarr-specs.readthedocs.io/en/latest',
        ('intersphinx_inv/zarr-specs.inv', None),
    ),
    'numpy': (
        'https://numpy.org/doc/stable/',
        ('intersphinx_inv/numpy.inv', None),
    ),
    'dask': (
        'https://docs.dask.org/en/latest/',
        ('intersphinx_inv/dask.inv', None),
    ),
}

rst_prolog = """
.. role:: python(code)
   :language: python
   :class: highlight

.. role:: cpp(code)
   :language: cpp
   :class: highlight

.. role:: json(code)
   :language: json
   :class: highlight
"""

# Warn about missing references
nitpicky = True

# The Sphinx C++ domain generates bogus undefined reference warnings for every
# C++ namespace that is mentioned in the documentation. All such namespaces need
# to be listed here in order to silence the warnings.
nitpick_ignore = [
    ('cpp:identifier', 'tensorstore'),
    ('cpp:identifier', '::tensorstore'),
    ('cpp:identifier', 'tensorstore::kvstore'),
    ('cpp:identifier', 'tensorstore::dtypes'),
    ('cpp:identifier', 'kvstore'),
    ('cpp:identifier', 'absl'),
    ('cpp:identifier', 'std'),
    ('cpp:identifier', '::std'),
    ('cpp:identifier', 'nlohmann'),
    ('cpp:identifier', '::nlohmann'),
    ('cpp:identifier', 'half_float'),
    ('cpp:identifier', '::half_float'),
]

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

object_description_options = []

json_schemas = [
    '*schema.yml',
    '**/*schema.yml',
]

json_schema_rst_prolog = """
.. default-role:: json:schema

.. default-literal-role:: json

.. highlight:: json
"""

python_apigen_modules = {
    'tensorstore': 'python/api/tensorstore.',
    'tensorstore.ocdbt': 'python/api/tensorstore.ocdbt.',
}

python_apigen_default_groups = [
    ('class:.*', 'Classes'),
    (r'.*:.*\.__(init|new)__', 'Constructors'),
    (r'.*:.*\.__eq__', 'Comparison operators'),
    (r'.*:.*\.__(str|repr)__', 'String representation'),
]

python_apigen_rst_prolog = """
.. default-role:: py:obj

.. default-literal-role:: python

.. highlight:: python

"""

python_module_names_to_strip_from_xrefs = [
    'tensorstore',
    'collections.abc',
    'numbers',
]

python_type_aliases = {
    'dtype': 'numpy.dtype',
    'Real': 'numbers.Real',
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
    'numpy.typing.ArrayLike': TypeXrefTarget(
        'py', 'data', 'numpy.typing.ArrayLike', 'ArrayLike'
    ),
    'tensorstore.RecheckCacheOption': TypeXrefTarget(
        'py', 'data', 'tensorstore.RecheckCacheOption', 'RecheckCacheOption'
    ),
    'NumpyIndexingSpec': TypeXrefTarget(
        'std', 'ref', 'python-numpy-style-indexing', 'NumpyIndexingSpec'
    ),
    'DimSelectionLike': TypeXrefTarget(
        'std', 'ref', 'python-dim-selections', 'DimSelectionLike'
    ),
    'DownsampleMethod': TypeXrefTarget(
        'json', 'schema', 'DownsampleMethod', 'DownsampleMethod'
    ),
}


def _monkey_patch_type_to_xref():
  _orig_python_type_to_xref = sphinx.domains.python.type_to_xref

  def _python_type_to_xref(
      target: str,
      env: Optional[sphinx.environment.BuildEnvironment] = None,
      *args,
      **kwargs,
  ) -> sphinx.addnodes.pending_xref:
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
    return _orig_python_type_to_xref(target, env, *args, **kwargs)

  for modname in [
      'sphinx.domains.python',
      # In newer sphinx versions, `type_to_xref` is actually defined in
      # `sphinx.domains.python._annotations`, and must be overridden there as
      # well.
      'sphinx.domains.python._annotations',
  ]:
    module = sys.modules.get(modname)
    if module is None:
      continue
    if getattr(module, 'type_to_xref', None) is _orig_python_type_to_xref:
      setattr(module, 'type_to_xref', _python_type_to_xref)


_monkey_patch_type_to_xref()

external_cpp_references = {
    'nlohmann::json': {
        'url': 'https://json.nlohmann.me/api/json/',
        'object_type': 'type alias',
        'desc': 'C++ type alias',
    },
    'nlohmann::basic_json': {
        'url': 'https://json.nlohmann.me/api/basic_json/',
        'object_type': 'class',
        'desc': 'C++ class',
    },
    'half_float::half': {
        'url': 'http://half.sourceforge.net/classhalf__float_1_1half.html',
        'object_type': 'class',
        'desc': 'C++ class',
    },
    'absl::Status': {
        'url': 'https://abseil.io/docs/cpp/guides/status',
        'object_type': 'class',
        'desc': 'C++ class',
    },
    'absl::StatusOr': {
        'url': 'https://abseil.io/docs/cpp/guides/statuss#returning-a-status-or-a-value',
        'object_type': 'class',
        'desc': 'C++ class',
    },
    'absl::OkStatus': {
        'url': 'https://abseil.io/docs/cpp/guides/status',
        'object_type': 'function',
        'desc': 'C++ function',
    },
    'absl::StatusCode': {
        'url': 'https://abseil.io/docs/cpp/guides/status-codes',
        'object_type': 'enum',
        'desc': 'C++ enumeration',
    },
    'absl::Time': {
        'url': 'https://abseil.io/docs/cpp/guides/time#absolute-times-with-absltime',
        'object_type': 'class',
        'desc': 'C++ class',
    },
    'absl::InfiniteFuture': {
        'url': 'https://abseil.io/docs/cpp/guides/time#absolute-times-with-absltime',
        'object_type': 'function',
        'desc': 'C++ function',
    },
    'absl::InfinitePast': {
        'url': 'https://abseil.io/docs/cpp/guides/time#absolute-times-with-absltime',
        'object_type': 'function',
        'desc': 'C++ function',
    },
    'absl::Now': {
        'url': 'https://abseil.io/docs/cpp/guides/time#absolute-times-with-absltime',
        'object_type': 'function',
        'desc': 'C++ function',
    },
    'absl::Duration': {
        'url': 'https://abseil.io/docs/cpp/guides/time#time-durations',
        'object_type': 'class',
        'desc': 'C++ class',
    },
    'absl::Cord': {
        'url': 'https://github.com/abseil/abseil-cpp/blob/master/absl/strings/cord.h',
        'object_type': 'class',
        'desc': 'C++ class',
    },
    'absl::AnyInvocable': {
        'url': 'https://github.com/abseil/abseil-cpp/blob/master/absl/functional/any_invocable.h',
        'object_type': 'class',
        'desc': 'C++ class',
    },
}

for code in [
    'kOk',
    'kCancelled',
    'kUnknown',
    'kInvalidArgument',
    'kNotFound',
    'kAlreadyExists',
    'kPermissionDenied',
    'kResourceExhausted',
    'kFailedPrecondition',
    'kAborted',
    'kOutOfRange',
    'kUnimplemented',
    'kInternal',
    'kUnavailable',
    'kDataLoss',
    'kUnauthenticated',
]:
  external_cpp_references[f'absl::StatusCode::{code}'] = {
      'url': 'https://abseil.io/docs/cpp/guides/status-codes',
      'object_type': 'enumerator',
      'desc': 'C++ enumerator',
  }

html_wrap_signatures_with_css = ['py']

object_description_options.append((
    '(cpp|c):.*',
    dict(
        clang_format_style={
            'BasedOnStyle': 'Google',
            'AlignAfterOpenBracket': 'Align',
            'AlignOperands': 'AlignAfterOperator',
            'AllowAllArgumentsOnNextLine': 'true',
            'AllowAllParametersOfDeclarationOnNextLine': 'false',
            'AlwaysBreakTemplateDeclarations': 'Yes',
            'BinPackArguments': 'true',
            'BinPackParameters': 'false',
            'BreakInheritanceList': 'BeforeColon',
            'ColumnLimit': '70',
            'ContinuationIndentWidth': '4',
            'Cpp11BracedListStyle': 'true',
            'DerivePointerAlignment': 'false',
            'IndentRequiresClause': 'true',
            'IndentWidth': '2',
            'IndentWrappedFunctionNames': 'false',
            'InsertBraces': 'false',
            'InsertTrailingCommas': 'None',
            'PointerAlignment': 'Left',
            'QualifierAlignment': 'Leave',
            'ReferenceAlignment': 'Pointer',
            'RemoveBracesLLVM': 'false',
            'RequiresClausePosition': 'OwnLine',
            'Standard': 'c++20',
            'PenaltyReturnTypeOnItsOwnLine': '1',
            'PenaltyBreakBeforeFirstCallParameter': '2',
            'PenaltyBreakAssignment': '3',
            'SpaceBeforeParens': 'Custom',
            'SpaceBeforeParensOptions': {
                'AfterRequiresInClause': 'true',
            },
        }
    ),
))

clang_format_command = os.environ.get('SPHINX_CLANG_FORMAT', 'clang-format')

cpp_strip_namespaces_from_signatures = ['tensorstore']

cpp_apigen_configs = [
    {
        'document_prefix': 'cpp/api/',
        # Generated by generate_cpp_api.py
        'api_data': 'cpp_api.json',
    },
]

cpp_apigen_rst_prolog = """
.. default-role:: cpp:expr

.. default-literal-role:: cpp

.. highlight:: cpp

"""


# Workaround for Sphinx parallel build inefficiency with large number of
# documents.
#
# This ensures that there is one worker per batch, and all workers are forked
# immediately at the start of reading/writing documents, such that the forked
# BuildEnvironment in each worker does not contain any partial results from
# previously-finished batches.
#
# https://github.com/sphinx-doc/sphinx/issues/10967
def _monkey_patch_parallel_maxbatch():
  orig_make_chunks = sphinx.util.parallel.make_chunks

  orig_add_task = sphinx.util.parallel.ParallelTasks.add_task
  orig_join_one = sphinx.util.parallel.ParallelTasks._join_one
  orig_join = sphinx.util.parallel.ParallelTasks.join

  def add_task(self, *args, **kwargs):
    try:
      self._in_add_task = True
      return orig_add_task(self, *args, **kwargs)
    finally:
      self._in_add_task = False

  sphinx.util.parallel.ParallelTasks.add_task = add_task

  def _join_one(self) -> bool:
    if getattr(self, '_in_add_task', False):
      return False
    return orig_join_one(self)

  def join(self):
    orig_join_one(self)
    return orig_join(self)

  sphinx.util.parallel.ParallelTasks.join = join

  sphinx.util.parallel.ParallelTasks._join_one = _join_one

  def make_chunks(arguments, nproc: int, maxbatch: int = 10000000):
    chunks = orig_make_chunks(arguments, nproc - 1, maxbatch)
    return chunks

  for modname in [
      'sphinx.util.parallel',
      'sphinx.builders',
  ]:
    module = importlib.import_module(modname)
    if getattr(module, 'make_chunks', None) is orig_make_chunks:
      setattr(module, 'make_chunks', make_chunks)


_monkey_patch_parallel_maxbatch()


def setup(app):
  # Exclude pybind11-builtin base class when displaying base classes.
  #
  # This base class is purely an implementation detail and not helpful to
  # display to users.
  def _autodoc_process_bases(app, name, obj, options, bases):
    bases[:] = [
        base for base in bases if base.__module__ != 'pybind11_builtins'
    ]

  app.connect('autodoc-process-bases', _autodoc_process_bases)
