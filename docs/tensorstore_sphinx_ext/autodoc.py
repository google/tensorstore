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
"""TensorStore customizations of sphinx.ext.autodoc."""

import functools
import re
import sys
from typing import List, Tuple, Any, Type, Optional

import docutils.nodes
import docutils.parsers.rst.states
import sphinx.addnodes
import sphinx.application
import sphinx.domains.python
import sphinx.environment
import sphinx.ext.autodoc
import sphinx.pycode.ast
from sphinx.pycode.ast import ast
import sphinx.util.docstrings
import sphinx.util.inspect
import sphinx.util.typing


def _remove_pymethod_self_type_annotation():
  """Monkey patches PyMethod to omit type annotation on `self` parameters.

  Also removes the return type annotation for `__init__`.

  pybind11 includes these annotations which just add verbosity.
  """
  PyMethod = sphinx.domains.python.PyMethod  # pylint: disable=invalid-name
  orig_handle_signature = PyMethod.handle_signature

  def handle_signature(
      self, sig: str,
      signode: sphinx.addnodes.desc_signature) -> Tuple[str, str]:
    result = orig_handle_signature(self, sig, signode)
    for param in signode.traverse(condition=sphinx.addnodes.desc_parameter):
      if param.children[0].astext() == 'self':
        # Remove any annotations on `self`
        del param.children[1:]
      break

    if signode['fullname'].endswith('.__init__'):
      # Remove first parameter.
      for param in signode.traverse(condition=sphinx.addnodes.desc_parameter):
        if param.children[0].astext() == 'self':
          param.parent.remove(param)
        break

      # Remove return type.
      for node in signode.traverse(condition=sphinx.addnodes.desc_returns):
        node.parent.remove(node)

    elif signode['fullname'].endswith('.__setitem__'):
      # Remove return type.
      for node in signode.traverse(condition=sphinx.addnodes.desc_returns):
        node.parent.remove(node)

    return result

  PyMethod.handle_signature = handle_signature


def _monkey_patch_py_xref_mixin():
  """Monkey patches PyXrefMixin to format types using `_parse_annotation`.

  This allows our custom `type_to_xref` implementation to be used.
  """
  PyXrefMixin = sphinx.domains.python.PyXrefMixin  # pylint: disable=invalid-name

  def make_xrefs(
      self, rolename: str, domain: str, target: str,
      innernode: Type[docutils.nodes.TextElement] = docutils.nodes.emphasis,
      contnode: Optional[docutils.nodes.Node] = None,
      env: Optional[sphinx.environment.BuildEnvironment] = None,
      inliner: Optional[docutils.parsers.rst.states.Inliner] = None,
      location: Optional[docutils.nodes.Node] = None,
  ) -> List[docutils.nodes.Node]:
    del self
    del rolename
    del domain
    del innernode
    del contnode
    del inliner
    del location
    return sphinx.domains.python._parse_annotation(target, env)  # pylint: disable=protected-access

  PyXrefMixin.make_xrefs = make_xrefs


def _process_docstring(app: sphinx.application.Sphinx, what: str, name: str,
                       obj: Any, options: Any, lines: List[str]) -> None:  # pylint: disable=g-doc-args
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
  del what
  del obj
  del options
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
  except:  # pylint: disable=bare-except
    # ignore errors
    return

  for param in sig.parameters.values():
    if param.annotation is not param.empty:
      insert_type(param.name, sphinx.util.typing.stringify(param.annotation))

  # TODO(jbms): Disable inserting rtype for now, since sphinx does display it
  # very nicely anyway.

  # def insert_rtype(rtype: str) -> None:
  #   rtype_pattern = '^:rtype: '
  #   # Skip insert if there is already an :rtype: field.
  #   if any(re.match(rtype_pattern, line) for line in lines):
  #     return

  #   # Only add :rtype: field if thre is a :returns: field
  #   returns_pattern = '^:returns?: '
  #   for i, line in enumerate(lines):
  #     if re.match(returns_pattern, line):
  #       lines.insert(i, f':rtype: {rtype}')
  #       break

  # if documenter.retann:
  #   insert_rtype(documenter.retann)


def _type_to_xref(
    text: str,
    env: sphinx.environment.BuildEnvironment) -> sphinx.addnodes.pending_xref:
  """Tensorstore-customized version of sphinx.domains.python.type_to_xref.

  This handles certain aliases and short names used in TensorStore type
  annotations.

  Args:
    text: "Type" referenced in the annotation.
    env: Sphinx build environment.

  Returns:
    Reference node.
  """
  reftarget = text
  refdomain = 'py'
  reftype = 'obj'
  if text in ('Optional', 'List', 'Union', 'Dict', 'Any', 'Iterator', 'Tuple',
              'Literal', 'Sequence', 'Callable'):
    reftarget = 'typing.' + text
  elif text in ('Real',):
    reftarget = 'numbers.Real'
  elif text == 'array':
    reftarget = 'numpy.ndarray'
  elif text == 'dtype':
    reftarget = 'numpy.dtype'
  elif text == 'array_like':
    reftarget = 'numpy:array_like'
    refdomain = 'std'
    reftype = 'any'
  elif text == 'NumpyIndexingSpec':
    reftarget = 'python-numpy-style-indexing'
    refdomain = 'std'
    reftype = 'ref'
  elif text == 'DimSelectionLike':
    reftarget = 'python-dim-selections'
    refdomain = 'std'
    reftype = 'ref'
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


def _transform_annotation_ast(tree: ast.AST) -> ast.AST:
  """Transforms the AST of a type annotation to improve the display.

  - Converts Union/Optional/Literal to "|" syntax allowed by PEP 604.  This
    syntax is not actually supported until Python 3.10 but displays better in
    the documentation.

  Args:
    tree: Original AST.

  Returns:
    Transformed AST.
  """

  class Transformer(ast.NodeTransformer):  # pylint: disable=missing-class-docstring

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
      if not isinstance(node.value, ast.Name):
        return node
      if node.value.id in ('Optional', 'Union', 'Literal'):
        elts = node.slice
        if isinstance(node.slice, ast.Index):
          elts = elts.value
        if isinstance(elts, ast.Tuple):
          elts = elts.elts
        else:
          elts = [node.slice]
        elts = [_transform_annotation_ast(x) for x in elts]
        if node.value.id == 'Optional':
          elts.append(ast.Constant(value=None))
        elif node.value.id == 'Literal':
          elts = [ast.Subscript(node.value, x, node.ctx) for x in elts]
        result = functools.reduce(
            (lambda left, right: ast.BinOp(left, ast.BitOr(), right)),
            elts)
        return result
      return ast.Subscript(node.value, _transform_annotation_ast(node.slice),
                           node.ctx)

  return ast.fix_missing_locations(Transformer().visit(tree))


def _parse_annotation(
    annotation: str,
    env: sphinx.environment.BuildEnvironment) -> List[docutils.nodes.Node]:
  """Parse type annotation."""

  # This is copied from Sphinx version 4, in order to support the type union
  # operator `|`.  We can eliminate most of this once we depend on Sphinx
  # version 4.
  #
  # It is subject to the following license notice:
  #
  # Copyright (c) 2007-2020 by the Sphinx team (see AUTHORS file).
  # All rights reserved.
  #
  # Redistribution and use in source and binary forms, with or without
  # modification, are permitted provided that the following conditions are
  # met:
  #
  # * Redistributions of source code must retain the above copyright
  #   notice, this list of conditions and the following disclaimer.
  #
  # * Redistributions in binary form must reproduce the above copyright
  #   notice, this list of conditions and the following disclaimer in the
  #   documentation and/or other materials provided with the distribution.
  #
  # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  # "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  # LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  # A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  # HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  # SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  # LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  # DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  # THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  def unparse(node: ast.AST) -> List[docutils.nodes.Node]:
    if isinstance(node, ast.Attribute):
      return [
          docutils.nodes.Text('%s.%s' % (unparse(node.value)[0], node.attr))
      ]
    elif isinstance(node, ast.BinOp):
      result: List[docutils.nodes.Node] = unparse(node.left)
      result.extend(unparse(node.op))
      result.extend(unparse(node.right))
      return result
    elif isinstance(node, ast.BitOr):
      return [
          docutils.nodes.Text(' '),
          sphinx.addnodes.desc_sig_punctuation('', '|'),
          docutils.nodes.Text(' ')
      ]
    elif isinstance(node, ast.Constant):  # type: ignore
      if node.value is Ellipsis:
        return [sphinx.addnodes.desc_sig_punctuation('', '...')]
      if node.value is None:
        return [
            docutils.nodes.literal(text='None', classes=['code', 'python'],
                                   language='python')
        ]
      else:
        return [docutils.nodes.Text(node.value)]
    elif isinstance(node, ast.Expr):
      return unparse(node.value)
    elif isinstance(node, ast.Index):
      return unparse(node.value)
    elif isinstance(node, ast.List):
      result = [sphinx.addnodes.desc_sig_punctuation('', '[')]
      for elem in node.elts:
        result.extend(unparse(elem))
        result.append(sphinx.addnodes.desc_sig_punctuation('', ', '))
      result.pop()
      result.append(sphinx.addnodes.desc_sig_punctuation('', ']'))
      return result
    elif isinstance(node, ast.Module):
      return sum((unparse(e) for e in node.body), [])
    elif isinstance(node, ast.Name):
      return [docutils.nodes.Text(node.id)]
    elif isinstance(node, ast.Subscript):
      if isinstance(node.value, ast.Name) and node.value.id == 'Literal':
        constant = node.slice
        if isinstance(constant, ast.Index):
          constant = constant.value
        if isinstance(constant, (ast.Constant, ast.Str)):
          if isinstance(constant, ast.Constant):
            value = constant.value
          else:
            value = constant.s
          return [
              docutils.nodes.literal(text=repr(value),
                                     classes=['code',
                                              'python'], language='python')
          ]
      result = unparse(node.value)
      result.append(sphinx.addnodes.desc_sig_punctuation('', '['))
      result.extend(unparse(node.slice))
      result.append(sphinx.addnodes.desc_sig_punctuation('', ']'))
      return result
    elif isinstance(node, ast.Tuple):
      if node.elts:
        result = []
        for elem in node.elts:
          result.extend(unparse(elem))
          result.append(sphinx.addnodes.desc_sig_punctuation('', ', '))
        result.pop()
      else:
        result = [
            sphinx.addnodes.desc_sig_punctuation('', '('),
            sphinx.addnodes.desc_sig_punctuation('', ')')
        ]

      return result
    else:
      if sys.version_info < (3, 8):
        if isinstance(node, ast.Ellipsis):
          return [sphinx.addnodes.desc_sig_punctuation('', '...')]
        elif isinstance(node, ast.NameConstant):
          return [docutils.nodes.Text(node.value)]

      raise SyntaxError  # unsupported syntax

  try:
    tree = sphinx.pycode.ast.parse(annotation)
    tree = _transform_annotation_ast(tree)
    result = unparse(tree)
    for i, node in enumerate(result):
      if isinstance(node, docutils.nodes.Text) and node.strip():
        result[i] = _type_to_xref(str(node), env)
  except SyntaxError:
    result = [_type_to_xref(annotation, env)]
  return [sphinx.addnodes.desc_type('', '', *result)]


def _monkey_patch_python_type_to_xref():
  """Modifies type_to_xref to support TensorStore-specific aliases."""

  sphinx.domains.python.type_to_xref = _type_to_xref
  sphinx.domains.python._parse_annotation = _parse_annotation  # pylint: disable=protected-access


def setup(app):  # pylint: disable=missing-function-docstring
  _monkey_patch_py_xref_mixin()
  _monkey_patch_python_type_to_xref()
  _remove_pymethod_self_type_annotation()

  # Must register `sphinx.ext.napoleon` first, since the `_process_docstring`
  # handler for the `autodoc-process-docstring` event needs to run after the
  # handler registered by `sphinx.ext.napoleon`.
  app.setup_extension('sphinx.ext.napoleon')
  app.connect('autodoc-process-docstring', _process_docstring)
  return {'parallel_read_safe': True, 'parallel_write_safe': True}
