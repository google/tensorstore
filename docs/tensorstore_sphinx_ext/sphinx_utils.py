# Copyright 2021 The TensorStore Authors
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
"""Utilities for use with Sphinx."""

import io
from typing import Optional, Dict, Union, List

import docutils.nodes
import docutils.parsers.rst.states
import docutils.statemachine
import sphinx.util.docutils


def to_statemachine_stringlist(
    content: str, source_path: str,
    source_line: int = 0) -> docutils.statemachine.StringList:
  """Converts to a docutils StringList with associated source info.

  All lines of `content` are assigned the same source info.

  Args:
    content: Source text.
    source_path: Path to the source file, for error messages.
    source_line: Line number in source file, for error messages.

  Returns:
    The string list, which may be passed to `nested_parse`.
  """
  list_lines = docutils.statemachine.string2lines(content)
  items = [(source_path, source_line)] * len(list_lines)
  return docutils.statemachine.StringList(list_lines, items=items)


def format_directive(
    name: str, *args: str, content: Optional[str] = None,
    options: Optional[Dict[str, Union[None, str, bool]]] = None) -> str:
  """Formats a RST directive into RST syntax.

  Args:
    name: Directive name, e.g. "json:schema".
    *args: List of directive arguments.
    content: Directive body content.
    options: Directive options.

  Returns:
    The formatted directive.
  """
  out = io.StringIO()
  out.write('\n\n')
  out.write(f'.. {name}::')
  for arg in args:
    out.write(f' {arg}')
  out.write('\n')
  if options:
    for key, value in options.items():
      if value is False or value is None:  # pylint: disable=g-bool-id-comparison
        continue
      if value is True:  # pylint: disable=g-bool-id-comparison
        value = ''
      out.write(f'   :{key}: {value}\n')
  if content:
    out.write('\n')
    for line in content.splitlines():
      out.write(f'   {line}\n')
  out.write('\n')
  return out.getvalue()


def parse_rst(state: docutils.parsers.rst.states.RSTState, text: str,
              source_path: str,
              source_line: int = 0) -> List[docutils.nodes.Node]:
  content = to_statemachine_stringlist(text, source_path, source_line)
  with sphinx.util.docutils.switch_source_input(state, content):
    node = docutils.nodes.container()
    # necessary so that the child nodes get the right source/line set
    node.document = state.document
    state.nested_parse(content, 0, node)
  return node.children


def summarize_element_text(node: docutils.nodes.Element) -> str:
  """Extracts a short text synopsis, e.g. for use as a tooltip."""

  # Recurisvely extract first paragraph
  while True:
    for p in node.traverse(condition=docutils.nodes.paragraph):
      if p is node:
        continue
      node = p
      break
    else:
      break

  text = node.astext()
  sentence_end = text.find('. ')
  if sentence_end != -1:
    text = text[:sentence_end + 1]
  return text
