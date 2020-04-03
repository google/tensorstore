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

import ast
import re
from typing import List, Tuple

import astor
import docutils.nodes
import sphinx.addnodes
import sphinx.domains.python
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


def _make_python_type_ref(target):
  reftarget = target
  if target == 'Optional' or target == 'List' or target == 'Union' or target == 'Dict':
    reftarget = 'typing.' + target
  elif target == 'Future' or target == 'Index':
    reftarget = 'tensorstore.' + target
  elif target == 'array':
    reftarget = 'numpy.ndarray'
  elif target == 'dtype':
    reftarget = 'numpy.dtype'
  prefix = 'tensorstore.'
  if target.startswith(prefix):
    target = target[len(prefix):]
  tnode = sphinx.addnodes.desc_type(target, target)
  pnode = sphinx.addnodes.pending_xref(
      '',
      refdomain='py',
      reftype='obj',
      reftarget=reftarget,
      refwarn=True,
  )
  pnode += tnode
  return pnode


def _make_python_type_ref_from_annotation(a: ast.AST):
  if isinstance(a, ast.Subscript):
    combined_node = sphinx.addnodes.desc_type()
    combined_node += _make_python_type_ref_from_annotation(a.value)
    combined_node += docutils.nodes.emphasis('[', '[')
    slice_value = a.slice.value
    if isinstance(slice_value, ast.Tuple):
      slice_elts = slice_value.elts
    else:
      slice_elts = [slice_value]
    for i, elt in enumerate(slice_elts):
      if i != 0:
        combined_node += docutils.nodes.emphasis(',', ',')
      combined_node += _make_python_type_ref_from_annotation(elt)
    combined_node += docutils.nodes.emphasis(']', ']')
    return combined_node
  return _make_python_type_ref(astor.to_source(a).strip())


def _render_python_arglist(signode: sphinx.addnodes.desc_signature,
                           arglist: str) -> None:
  paramlist = sphinx.addnodes.desc_parameterlist()
  args_ast = ast.parse('def f(' + arglist + '): pass').body[0].args

  def do_arg(arg, prefix=''):
    nonlocal paramlist
    param = sphinx.addnodes.desc_parameter('', '', noemph=True)
    arg_name_plain = prefix + arg.arg
    arg_name_output = prefix + arg.arg
    if arg.annotation:
      arg_name_plain += ': '
      arg_name_output += ':\xa0'
    param += sphinx.addnodes.literal_emphasis(arg_name_plain, arg_name_output)
    if arg.annotation:
      param += _make_python_type_ref_from_annotation(arg.annotation)
    paramlist += param

  for arg in args_ast.args:
    do_arg(arg)
  if args_ast.vararg:
    do_arg(args_ast.vararg, '*')
  if args_ast.kwonlyargs:
    paramlist += sphinx.addnodes.literal_emphasis('*', '*')
    for arg in args_ast.kwonlyargs:
      do_arg(arg)
  if args_ast.kwarg:
    do_arg(args_ast.kwarg, '**')
  signode += paramlist


sphinx.domains.python._pseudo_parse_arglist = _render_python_arglist  # pylint: disable=protected-access

old_desc_returns = sphinx.addnodes.desc_returns


def _python_desc_returns(part, unused):
  del unused
  rnode = old_desc_returns()
  rnode += _make_python_type_ref_from_annotation(ast.parse(part).body[0].value)
  return rnode


_old_handle_signature = sphinx.domains.python.PyObject.handle_signature


def _handle_python_signature(self, sig: str,
                             signode: sphinx.addnodes.desc_signature):
  cur_old_desc_returns = sphinx.addnodes.desc_returns
  sphinx.addnodes.desc_returns = _python_desc_returns
  result = _old_handle_signature(self, sig, signode)
  sphinx.addnodes.desc_returns = cur_old_desc_returns
  return result


sphinx.domains.python.PyObject.handle_signature = _handle_python_signature


def setup(app):
  app.setup_extension('sphinx.ext.autosummary')
  app.add_directive('autosummary', TensorstoreAutosummary)
  app.connect('autodoc-process-signature', get_autodoc_signature)
