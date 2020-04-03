#!/usr/bin/env python2
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
"""Generates MakeArray{,View} overloads for array.h."""

from __future__ import unicode_literals

import io
import os

import update_generated_source_code


def print_for_rank(rank, is_const, is_view, origin_type):
  """Print a single MakeArray or MakeArrayView definition.

  Args:
    rank: Positive int.  Specifies the rank of the array.
    is_const: bool.  If `True`, generates an overload for a const element type.
    is_view: bool.  If `True`, generates an overload that returns a view rather
      than a copy.
    origin_type: One of `'span'` (to generate an overload where the offset is
      specified as a `span`), `'array`` (to generate an overload where the
      offset is specified as an array, in order to allow it to be called using a
      braced list), or `None` (to generate zero-origin overload).

  Returns:
    The overload definition as a `str`.
  """
  maybe_const_element = 'const Element' if is_const else 'Element'
  is_offset = (origin_type is not None)
  function_name = 'OffsetArray' if is_offset else 'Array'
  if origin_type == 'span':
    origin_parameter = ('span<const Index, {rank}> origin, '.format(rank=rank))
  elif origin_type == 'array':
    origin_parameter = ('const Index (&origin)[OriginRank], '.format(rank=rank))
  else:
    origin_parameter = ''
  origin_kind_parameter = ', offset_origin' if is_offset else ''
  origin_argument = 'origin, ' if is_offset else ''
  out = io.StringIO()
  if is_view:
    deref_expr = '[0]' * rank
    if is_offset:
      start_ptr = ('AddByteOffset(ElementPointer<{element}>(&arr{deref_expr}), '
                   '-layout.origin_byte_offset())'.format(
                       element=maybe_const_element, deref_expr=deref_expr))
    else:
      start_ptr = '&arr{deref_expr}'.format(deref_expr=deref_expr)
  if is_view:
    out.write(r"""
/// Returns a rank-{rank} ArrayView that points to the specified C array.
///""".format(rank=rank))
    if is_const:
      out.write(r"""
/// This overload can be called with a braced list.
///""")
  else:
    out.write(r"""
/// Returns a rank-{rank} SharedArray containing a copy of the specified C array.
///""".format(rank=rank))
  if is_offset:
    out.write(r"""
/// \param origin The origin vector of the array.""")
  if is_view:
    out.write(r"""
/// \param arr The C array to which the returned `ArrayView` will point.
/// \remark The caller is responsible for ensuring that the returned array is
///     not used after `arr` becomes invalid.
""")
  else:
    out.write(r"""
/// \param arr The C array to be copied.
""")
  out.write('template <typename Element')
  for i in range(rank):
    out.write(', Index N%d' % i)
  if origin_type == 'array':
    out.write(', std::ptrdiff_t OriginRank')
  out.write('>\n')
  if is_view:
    out.write(
        'ArrayView<{element}, {rank}{origin_kind_parameter}> '
        'Make{function_name}View({origin_parameter}{element} (&arr)'.format(
            rank=rank,
            element=maybe_const_element,
            origin_parameter=origin_parameter,
            origin_kind_parameter=origin_kind_parameter,
            function_name=function_name))
  else:
    out.write('SharedArray<Element, {rank}{origin_kind_parameter}> '
              'Make{function_name}({origin_parameter}{element} (&arr)'.format(
                  rank=rank,
                  element=maybe_const_element,
                  origin_parameter=origin_parameter,
                  origin_kind_parameter=origin_kind_parameter,
                  function_name=function_name))
  for i in range(rank):
    out.write('[N%d]' % i)
  out.write(') {\n')
  if origin_type == 'array':
    out.write('static_assert(OriginRank == {rank}, '
              '"Origin vector must have length {rank}.");\n'.format(rank=rank))
  if is_view:
    out.write("""static constexpr Index shape[] = {""")
    out.write(', '.join('N%d' % i for i in range(rank)))
    out.write('};\n')
    out.write('static constexpr Index byte_strides[] = {')
    for i in range(rank):
      if i != 0:
        out.write(', ')
      out.write(''.join('N%d * ' % i for i in range(i + 1, rank)))
      out.write('sizeof(Element)')
    out.write("""};
    """)
    out.write("""StridedLayoutView<{rank}{origin_kind_parameter}> layout"""
              """({origin_argument}shape, byte_strides);
    """.format(rank=rank,
               origin_kind_parameter=origin_kind_parameter,
               origin_argument=origin_argument))
    out.write('return {{{start_ptr}, layout}};'.format(start_ptr=start_ptr))
  else:
    out.write(
        '  return MakeCopy(Make{function_name}View({origin_argument}arr));'
        .format(function_name=function_name, origin_argument=origin_argument))
  out.write('\n}\n')
  return out.getvalue()


def write_functions(out, ranks):
  for origin_type in [None, 'span', 'array']:
    for rank in ranks:
      for is_const in [False, True]:
        out.write(
            print_for_rank(
                rank, is_const=is_const, is_view=True, origin_type=origin_type))

    for rank in ranks:
      out.write((print_for_rank(
          rank, is_const=True, is_view=False, origin_type=origin_type)))


def main():
  max_rank = 6

  # Write the rank-1 overloads to array.h, and write the higher-rank overloads
  # to make_array.inc.
  out = io.StringIO()
  write_functions(out, [1])
  out.write("""

// Defines MakeArray, MakeArrayView, MakeOffsetAray, and MakeOffsetArrayView
// overloads for multi-dimensional arrays of rank 2 to {max_rank}.
#include "third_party/tensorstore/make_array.inc"
""".format(max_rank=max_rank))

  update_generated_source_code.update_generated_content(
      path=os.path.join(os.path.dirname(__file__), 'array.h'),
      script_name=os.path.basename(__file__),
      new_content=out.getvalue(),
  )

  out = io.StringIO()
  write_functions(out, range(2, max_rank + 1))
  update_generated_source_code.update_generated_content(
      path=os.path.join(os.path.dirname(__file__), 'make_array.inc'),
      script_name=os.path.basename(__file__),
      new_content=out.getvalue(),
  )


if __name__ == '__main__':
  main()
