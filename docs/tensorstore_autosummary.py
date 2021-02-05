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
"""TensorStore-specific Python documentation generation.

This extension overrides the built-in autosummary directive to automatically
include module members, as described here:
https://stackoverflow.com/questions/20569011/python-sphinx-autosummary-automated-listing-of-member-functions

Additionally, this modifies autosummary and autodoc to handle pybind11
overloaded functions.
"""

import re
from typing import List, Tuple

import docutils.nodes
import sphinx.addnodes
import sphinx.domains.python
import sphinx.environment
import sphinx.ext.autodoc
from sphinx.ext.autosummary import Autosummary
from sphinx.ext.autosummary import extract_summary
from sphinx.ext.autosummary import import_by_name
from sphinx.ext.autosummary import mangle_signature
from sphinx.util.docstrings import prepare_docstring


def _parse_overloaded_function_docstring(doc: str):
  m = re.match('^([^(]+)\\(', doc)
  display_name = m.group(1)
  overloaded_prefix = '\nOverloaded function.\n'
  doc = doc[doc.index(overloaded_prefix) + len(overloaded_prefix):]
  i = 1

  def get_prefix(i: int):
    return '\n%d. %s(' % (i, display_name)

  prefix = get_prefix(i)
  parts = []  # type: List[Tuple[str, str]]
  while doc:
    if not doc.startswith(prefix):
      raise RuntimeError('Docstring does not contain %r as expected: %r' % (
          prefix,
          doc,
      ))
    doc = doc[len(prefix) - 1:]
    nl_index = doc.index('\n')
    part_sig = doc[:nl_index]
    doc = doc[nl_index + 1:]
    i += 1
    prefix = get_prefix(i)
    end_index = doc.find(prefix)
    if end_index == -1:
      part = doc
      doc = ''
    else:
      part = doc[:end_index]
      doc = doc[end_index:]
    parts.append((part_sig, part))
  return parts


def _get_attribute_type(obj: property):
  if obj.fget is not None:
    doc = obj.fget.__doc__
    if doc is not None:
      lines = doc.splitlines()
      match = re.fullmatch('^(\\(.*)\\)\\s*->\\s*(.*)$', lines[0])
      if match:
        args, retann = match.groups()
        del args
        return retann
  return None


class TensorstoreAutosummary(Autosummary):

  def get_items(self, names: List[str]) -> List[Tuple[str, str, str, str]]:
    items = []  # type: List[Tuple[str, str, str, str]]
    for display_name, sig, summary, real_name in super().get_items(names):
      if summary == 'Initialize self.':
        continue
      real_name, obj, parent, modname = import_by_name(real_name)
      del parent
      del modname
      if summary == 'Overloaded function.':
        for part_sig, part in _parse_overloaded_function_docstring(obj.__doc__):
          max_item_chars = 50
          max_chars = max(10, max_item_chars - len(display_name))
          mangled_sig = mangle_signature(part_sig, max_chars=max_chars)
          part_summary = extract_summary(part.splitlines(), self.state.document)
          items.append((display_name, mangled_sig, part_summary, real_name))
      else:
        if isinstance(obj, property):
          retann = _get_attribute_type(obj)
          if retann is not None:
            sig = ': ' + retann

        items.append((display_name, sig, summary, real_name))
    return items

  def run(self):
    result = super().run()

    # Strip out duplicate toc entries due to overloaded functions
    toc_node = result[-1].children[0]
    seen_docnames = set()
    new_entries = []
    if 'entries' in toc_node:
      for _, docn in toc_node['entries']:
        if docn in seen_docnames:
          continue
        seen_docnames.add(docn)
        new_entries.append((None, docn))
      toc_node['entries'] = new_entries

    return result


def _overloaded_function_generate_documentation(self, old_documenter, *args,
                                                **kwargs):
  if not self.parse_name():
    return
  if not self.import_object():
    return
  doc = self.object.__doc__
  if doc is None or '\nOverloaded function.\n\n' not in doc:
    old_documenter(self, *args, **kwargs)
  else:
    old_indent = self.indent
    for part_sig, part_doc in _parse_overloaded_function_docstring(doc):
      tab_width = self.directive.state.document.settings.tab_width
      full_part_doc = '%s%s\n%s' % (self.object.__name__, part_sig, part_doc)
      self._new_docstrings = [prepare_docstring(full_part_doc, 1, tab_width)]  # pylint: disable=protected-access
      self.indent = old_indent
      old_documenter(self, *args, **kwargs)
      self.options.noindex = True


orig_autodoc_function_documenter_generate = sphinx.ext.autodoc.FunctionDocumenter.generate


def _autodoc_function_documenter_generate(self, *args, **kwargs):
  return _overloaded_function_generate_documentation(
      self, orig_autodoc_function_documenter_generate, *args, **kwargs)


sphinx.ext.autodoc.FunctionDocumenter.generate = _autodoc_function_documenter_generate

orig_autodoc_method_documenter_generate = sphinx.ext.autodoc.MethodDocumenter.generate


def _autodoc_method_documenter_generate(self, *args, **kwargs):
  return _overloaded_function_generate_documentation(
      self, orig_autodoc_method_documenter_generate, *args, **kwargs)


sphinx.ext.autodoc.MethodDocumenter.generate = _autodoc_method_documenter_generate


def get_autodoc_signature(app, what, name, obj, options, signature,
                          return_annotation):
  del app
  del what
  del name
  del obj
  del options

  return signature, return_annotation


# Modified version of `sphinx.domains.python.type_to_xref` to handle
# TensorStore-specific aliases.
#
# This is monkey-patched in below.
def type_to_xref(
    text: str,
    env: sphinx.environment.BuildEnvironment) -> sphinx.addnodes.pending_xref:
  reftarget = text
  refdomain = 'py'
  reftype = 'obj'
  if text in ('Optional', 'List', 'Union', 'Dict'):
    reftarget = 'typing.' + text
  elif text == 'array':
    reftarget = 'numpy.ndarray'
  elif text == 'dtype':
    reftarget = 'numpy.dtype'
  elif text == 'array_like':
    reftarget = 'numpy:array_like'
    refdomain = 'std'
    reftype = 'any'
  elif text == 'DownsampleMethod':
    reftarget = 'json-schema-https://github.com/google/tensorstore/json-schema/driver/downsample#method'
    refdomain = 'std'
    reftype = 'ref'
  prefix = 'tensorstore.'
  if text.startswith(prefix):
    text = text[len(prefix):]
  if env:
    kwargs = {
        'py:module': env.ref_context.get('py:module'),
        'py:class': env.ref_context.get('py:class')
    }
  else:
    kwargs = {}

  return sphinx.addnodes.pending_xref(
      '',
      docutils.nodes.Text(text),
      refdomain=refdomain,
      reftype=reftype,
      reftarget=reftarget,
      refwarn=True,
      refexplicit=True,
      **kwargs)


# Monkey-patch in modified `type_to_xref` implementation.
sphinx.domains.python.type_to_xref = type_to_xref


def setup(app):
  app.setup_extension('sphinx.ext.autosummary')
  app.add_directive('autosummary', TensorstoreAutosummary)
  app.connect('autodoc-process-signature', get_autodoc_signature)
