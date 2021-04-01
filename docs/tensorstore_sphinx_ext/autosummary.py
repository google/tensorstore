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
from typing import List, Tuple, Any, Optional, Type, Union

import docutils.nodes
import sphinx.addnodes
import sphinx.application
import sphinx.domains.python
import sphinx.environment
import sphinx.ext.autodoc
import sphinx.ext.autosummary
import sphinx.util.docstrings
import sphinx.util.docutils
import sphinx.util.inspect
import sphinx.util.typing


def _parse_overloaded_function_docstring(doc: str):
  m = re.match('^([^(]+)\\(', doc)
  if m is None:
    raise ValueError(
        f'Failed to determine display name from docstring: {repr(doc)}')
  display_name = m.group(1)
  overloaded_prefix = '\nOverloaded function.\n'
  doc = doc[doc.index(overloaded_prefix) + len(overloaded_prefix):]
  i = 1

  def get_prefix(i: int):
    return '\n%d. %s(' % (i, display_name)

  prefix = get_prefix(i)
  parts: List[Tuple[str, str]] = []
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


class TensorstoreAutosummary(sphinx.ext.autosummary.Autosummary):

  def get_items(self, names: List[str]) -> List[Tuple[str, str, str, str]]:
    items: List[Tuple[str, str, str, str]] = []
    for display_name, sig, summary, real_name in super().get_items(names):
      if summary == 'Initialize self.':
        continue
      real_name, obj, parent, modname = sphinx.ext.autosummary.import_by_name(
          real_name)
      del parent
      del modname
      if summary == 'Overloaded function.':
        for part_sig, part in _parse_overloaded_function_docstring(obj.__doc__):
          max_item_chars = 50
          max_chars = max(10, max_item_chars - len(display_name))
          mangled_sig = sphinx.ext.autosummary.mangle_signature(
              part_sig, max_chars=max_chars)
          part_summary = sphinx.ext.autosummary.extract_summary(
              part.splitlines(), self.state.document)
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
    if result:
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


def _monkey_patch_autodoc_function_documenter(
    documenter_cls: Union[Type[sphinx.ext.autodoc.FunctionDocumenter],
                          Type[sphinx.ext.autodoc.MethodDocumenter]]):

  orig_generate = documenter_cls.generate

  def generate(self, *args, **kwargs):
    if not self.parse_name():
      return
    if not self.import_object():
      return

    current_documenter_map = self.env.temp_data.setdefault(
        'tensorstore_autodoc_current_documenter', {})
    current_documenter_map[self.fullname] = self

    doc = self.object.__doc__
    if doc is None or '\nOverloaded function.\n\n' not in doc:
      orig_generate(self, *args, **kwargs)
      return
    old_indent = self.indent

    for part_sig, part_doc in _parse_overloaded_function_docstring(doc):
      tab_width = self.directive.state.document.settings.tab_width
      full_part_doc = '%s%s\n%s' % (self.object.__name__, part_sig, part_doc)
      self._new_docstrings = [
          sphinx.util.docstrings.prepare_docstring(full_part_doc, 1, tab_width)
      ]  # pylint: disable=protected-access

      self.indent = old_indent
      orig_generate(self, *args, **kwargs)
      self.options.noindex = True

  documenter_cls.generate = generate


def _monkey_patch_py_xref_mixin():
  PyXrefMixin = sphinx.domains.python.PyXrefMixin

  def make_xrefs(
      self, rolename: str, domain: str, target: str,
      innernode: Type[docutils.nodes.TextElement] = docutils.nodes.emphasis,
      contnode: docutils.nodes.Node = None,
      env: sphinx.environment.BuildEnvironment = None
  ) -> List[docutils.nodes.Node]:
    return sphinx.domains.python._parse_annotation(target, env)

  PyXrefMixin.make_xrefs = make_xrefs


def _process_docstring(app: sphinx.application.Sphinx, what: str, name: str,
                       obj: Any, options: Any, lines: List[str]) -> None:
  """Adds `:type <param>: <type>` fields to docstrings based on annotations.

  This function is intended to be registered for the 'autodoc-process-docstring'
  signal, and must be registered *after* sphinx.ext.napoleon if Google-style
  docstrings are to be supported.

  Sphinx allows the type of a parameter to be indicated by a `:type <param>:`
  field.  However, in the TensorStore documentation all parameter types are
  instead indicated by annotations in the signature.

  Sphinx provides the `autodoc_typehints='description'` option which adds `:type
  <param>` fields based on the annotations.  However, it has a number of
  problematic limitations:

  - It only supports real annotations, stored in the `__annotations__` attribute
    of a function.  It isn't compatible with the `autodoc_docstring_signature`
    option, which we rely on for pybind11-defined functions and to support
    overloaded functions.

  - If you specify `autodoc_typehints='description'`, autodoc strips out
    annotations from the signature.  We would like to include them in both
    places.

  - It adds a `:type` field for all parameters, even ones that are not
    documented.

  This function providse the same functionality as the
  `autodoc_typehints='description'` option but without those limitations.
  """
  current_documenter_map = app.env.temp_data.get(
      'tensorstore_autodoc_current_documenter')
  if not current_documenter_map:
    return
  documenter = current_documenter_map.get(name)
  if not documenter:
    return

  def _get_field_pattern(field_names: List[str], param: str) -> str:
    return '^:(?:' + '|'.join(
        re.escape(name)
        for name in field_names) + r')\s+' + re.escape(param) + ':'

  def insert_type(param: str, typ: str) -> None:
    type_pattern = _get_field_pattern(['paramtype', 'type'], param)
    pattern = _get_field_pattern([
        'param', 'parameter', 'arg', 'argument', 'keyword', 'kwarg', 'kwparam'
    ], param)
    # First check if there is already a type field
    for line in lines:
      if re.match(type_pattern, line):
        # :type: field already present for `param`, don't add another one.
        return

    # Only add :type: field if `param` is documented.
    for i, line in enumerate(lines):
      if re.match(pattern, line):
        lines.insert(i, f':type {param}: {typ}')
        return

  if not documenter.args:
    return

  try:
    sig = sphinx.util.inspect.signature_from_str(
        f'func({documenter.args}) -> None')
  except:
    # ignore errors
    return

  for param in sig.parameters.values():
    if param.annotation is not param.empty:
      insert_type(param.name, sphinx.util.typing.stringify(param.annotation))


def _monkey_patch_python_type_to_xref():
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
      reftarget = 'DownsampleMethod'
      refdomain = 'json'
      reftype = 'schema'
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

    return sphinx.addnodes.pending_xref('', docutils.nodes.Text(text),
                                        refdomain=refdomain, reftype=reftype,
                                        reftarget=reftarget, refwarn=True,
                                        refexplicit=True, **kwargs)

  # Monkey-patch in modified `type_to_xref` implementation.
  sphinx.domains.python.type_to_xref = type_to_xref


def setup(app):
  _monkey_patch_autodoc_function_documenter(
      sphinx.ext.autodoc.FunctionDocumenter)
  _monkey_patch_autodoc_function_documenter(sphinx.ext.autodoc.MethodDocumenter)
  _monkey_patch_py_xref_mixin()
  _monkey_patch_python_type_to_xref()

  app.setup_extension('sphinx.ext.autosummary')
  # Must register `sphinx.ext.napoleon` first, since the `_process_docstring`
  # handler for the `autodoc-process-docstring` event needs to run after the
  # handler registered by `sphinx.ext.napoleon`.
  app.setup_extension('sphinx.ext.napoleon')
  app.add_directive('autosummary', TensorstoreAutosummary, override=True)
  app.connect('autodoc-process-docstring', _process_docstring)
  return {'parallel_read_safe': True, 'parallel_write_safe': True}
