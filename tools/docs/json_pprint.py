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
"""Adaptation of pprint.py from the Python standard library to JSON."""
#  Author:      Fred L. Drake, Jr.
#               fdrake@acm.org

import io
import json


def pformat(obj, indent=1, width=80, depth=None, *, compact=False):
  """Format a Python object into a pretty-printed representation."""
  return _PrettyPrinter(
      indent=indent, width=width, depth=depth, compact=compact).pformat(obj)


class _PrettyPrinter:
  """Implementation of `pformat`."""

  def __init__(self, indent=1, width=80, depth=None, *, compact=False):
    indent = int(indent)
    width = int(width)
    if indent < 0:
      raise ValueError("indent must be >= 0")
    if depth is not None and depth <= 0:
      raise ValueError("depth must be > 0")
    if not width:
      raise ValueError("width must be != 0")
    self._depth = depth
    self._indent_per_level = indent
    self._width = width
    self._compact = bool(compact)

  def pformat(self, obj):
    sio = io.StringIO()
    self._format(obj, sio, 0, 0)
    return sio.getvalue()

  def _repr(self, obj):
    return json.dumps(obj)

  def _format(self, obj, stream, indent, allowance):
    rep = self._repr(obj)
    max_width = self._width - indent - allowance
    if len(rep) > max_width:
      method = None
      if isinstance(obj, dict):
        method = self._pprint_dict
      elif isinstance(obj, list):
        method = self._pprint_list

      if method is not None:
        method(obj, stream, indent, allowance)
        return
    stream.write(rep)

  def _pprint_dict(self, obj, stream, indent, allowance):
    write = stream.write
    write("{")
    if self._indent_per_level > 1:
      write((self._indent_per_level - 1) * " ")
    length = len(obj)
    if length:
      self._format_dict_items(obj.items(), stream, indent, allowance + 1)
    write("}")

  def _pprint_list(self, obj, stream, indent, allowance):
    stream.write("[")
    self._format_items(obj, stream, indent, allowance + 1)
    stream.write("]")

  def _format_dict_items(self, items, stream, indent, allowance):
    write = stream.write
    indent += self._indent_per_level
    write("\n" + " " * indent)
    delimnl = ",\n" + " " * indent
    last_index = len(items) - 1
    for i, (key, ent) in enumerate(items):
      last = i == last_index
      rep = self._repr(key)
      write(rep)
      write(": ")
      self._format(ent, stream, indent, allowance if last else 1)
      if not last:
        write(delimnl)
      else:
        write("\n" + " " * (indent - self._indent_per_level))

  def _format_items(self, items, stream, indent, allowance):
    write = stream.write
    indent += self._indent_per_level
    if self._indent_per_level > 1:
      write((self._indent_per_level - 1) * " ")
    delimnl = ",\n" + " " * indent
    delim = ""
    width = max_width = self._width - indent + 1
    it = iter(items)
    try:
      next_ent = next(it)
    except StopIteration:
      return
    last = False
    while not last:
      ent = next_ent
      try:
        next_ent = next(it)
      except StopIteration:
        last = True
        max_width -= allowance
        width -= allowance
      if self._compact:
        rep = self._repr(ent)
        w = len(rep) + 2
        if width < w:
          width = max_width
          if delim:
            delim = delimnl
        if width >= w:
          width -= w
          write(delim)
          delim = ", "
          write(rep)
          continue
      write(delim)
      delim = delimnl
      self._format(ent, stream, indent, allowance if last else 1)
