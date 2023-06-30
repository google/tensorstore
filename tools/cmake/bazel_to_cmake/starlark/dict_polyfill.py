# Copyright 2022 The TensorStore Authors
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
"""Polyfill to change python dict semantics to Bazel dict semantics.

1. Bazel `dict` includes the union `operator|`, which was added by PEP 584
  (https://peps.python.org/pep-0584/) to Python 3.9.
  This module polyfills the operator unconditionally.

2. Bazel `dict`.`keys` returns a `list` rather than a python `dict_keys`

There are two components the polyfill:

- A `DictWithUnion` wrapper class that defines the new operators.  When the
  polyfill is applied, the global `dict` will refer to this wrapper class.

- An AST transformation that modifies dict literal and dict comprehension
  expressions to produce a `DictWithUnion` value rather than a built-in `dict`.
"""

import ast


class DictWithUnion(dict):
  """dict wrapper that adds union operator.

  bazel_globals.py defines two globals, `dict` and `__DictWithUnion`, that refer
  to this class.
  """

  def __or__(self, other):
    if not isinstance(other, dict):
      return NotImplemented
    new = DictWithUnion(self)
    new.update(other)
    return new

  def __ror__(self, other):
    if not isinstance(other, dict):
      return NotImplemented
    new = DictWithUnion(other)
    new.update(self)
    return new

  def __ior__(self, other):
    dict.update(self, other)
    return self

  def keys(self):
    # bazel dict.keys() returns list, not dict_keys.
    return list(super().keys())


class ASTTransformer(ast.NodeTransformer):
  """AST transformer that wraps dict literals and comprehension expressions.

  This is used by `_exec_module` in `evaluation.py` to apply the polyfill.
  """

  def visit_Dict(self, node):  # pylint: disable=invalid-name
    return self._wrap(node)

  def visit_DictComp(self, node):  # pylint: disable=invalid-name
    return self._wrap(node)

  def _wrap(self, node):
    # Note: Use an "obfuscated" alias for `dict` to prevent shadowing by a local
    # variable named `dict`.  This global is defined in `bazel_globals.py` as an
    # alias for the `DictWithUnion` class defined above.
    return ast.copy_location(
        ast.Call(
            ast.Name('__DictWithUnion', ctx=ast.Load()),
            args=[self.generic_visit(node)],
            keywords=[],
        ),
        node,
    )
