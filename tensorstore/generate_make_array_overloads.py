#!/usr/bin/env python3
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

import io
import os
from typing import List

import update_generated_source_code


def output_comment(out: io.StringIO, is_view: bool, origin_type: str) -> None:
  """Print comments for MakeArray or MakeArrayView definition.

  Args:
    out: Output stream.
    is_view: bool.  If `True`, generates an overload that returns a view rather
      than a copy.
    origin_type: One of `'span'` (to generate an overload where the offset is
      specified as a `span`), `'array`` (to generate an overload where the
      offset is specified as an array, in order to allow it to be called using a
      braced list), or `None` (to generate zero-origin overload).
  """
  if is_view:
    out.write(r"""
  /// Returns an `ArrayView` that points to the specified C array.""")
  else:
    out.write(r"""
  /// Returns a `SharedArray` containing a copy of the specified C array.""")
  out.write(r"""
  ///
  /// .. note::
  ///
  ///    Only the rank-1 and rank-2 overloads are shown, but C arrays with up to
  ///    6 dimensions are supported.
  ///""")
  if origin_type:
    out.write(r"""
  /// \param origin The origin vector of the array.  May be specified as a
  ///     braced list, e.g. `MakeOffsetArray({1, 2}, {{3, 4, 5}, {6, 7, 8}})`."""
             )
  if is_view:
    out.write(r"""
  /// \param array The C array to which the returned `ArrayView` will point.
  ///     May be specified as a (nested) braced list, e.g.
  ///     `MakeArrayView({{1, 2, 3}, {4, 5, 6}})`, in which case the inferred
  ///     `Element` type will be ``const``-qualified.
  ///
  /// .. warning::
  ///
  ///    The caller is responsible for ensuring that the returned array is
  ///    not used after `array` becomes invalid.
  ///
""")
  else:
    out.write(r"""
  /// \param array The C array to be copied.  May be specified as a (nested)
  ///     braced list, e.g. `MakeArray({{1, 2, 3}, {4, 5, 6}})`.
""")
  out.write(r"""/// \relates Array
  /// \membergroup Creation functions
  /// \id array
""")


def output_make_array(out: io.StringIO, rank: int, is_const: bool,
                      is_view: bool, origin_type: str) -> None:
  """Print a single MakeArray or MakeArrayView definition.

  Args:
    out: Output stream.
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
  if is_view:
    deref_expr = '[0]' * rank
    if is_offset:
      start_ptr = (
          'AddByteOffset(ElementPointer<{element}>(&array{deref_expr}), '
          '-layout.origin_byte_offset())'.format(element=maybe_const_element,
                                                 deref_expr=deref_expr))
    else:
      start_ptr = '&array{deref_expr}'.format(deref_expr=deref_expr)
  out.write('template <typename Element')
  for i in range(rank):
    out.write(', Index N%d' % i)
  if origin_type == 'array':
    out.write(', ptrdiff_t OriginRank')
  out.write('>\n')
  if is_view:
    out.write(
        'ArrayView<{element}, {rank}{origin_kind_parameter}> '
        'Make{function_name}View({origin_parameter}{element} (&array)'.format(
            rank=rank, element=maybe_const_element,
            origin_parameter=origin_parameter,
            origin_kind_parameter=origin_kind_parameter,
            function_name=function_name))
  else:
    out.write('SharedArray<Element, {rank}{origin_kind_parameter}> '
              'Make{function_name}({origin_parameter}{element} (&array)'.format(
                  rank=rank, element=maybe_const_element,
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
    """.format(rank=rank, origin_kind_parameter=origin_kind_parameter,
               origin_argument=origin_argument))
    out.write('return {{{start_ptr}, layout}};'.format(start_ptr=start_ptr))
  else:
    out.write(
        '  return MakeCopy(Make{function_name}View({origin_argument}array));'.
        format(function_name=function_name, origin_argument=origin_argument))
  out.write('\n}\n')


def write_functions(out: io.StringIO, ranks: List[int]) -> None:
  for origin_type in [None, 'span', 'array']:
    for is_view in [True, False]:
      for rank in ranks:

        # Only comment rank 1, non-array
        if origin_type != 'array':
          if rank == 1:
            output_comment(out, is_view, origin_type)
          elif rank > 2:
            out.write('\n')
        else:
          out.write('\n')

        for is_const in [False, True]:
          output_make_array(out=out, rank=rank, is_const=is_const,
                            is_view=is_view, origin_type=origin_type)


def main():
  max_rank = 6

  # Write the rank 1 and 2 overloads to array.h, and write the higher-rank
  # overloads to make_array.inc.
  out = io.StringIO()
  write_functions(out, [1, 2])
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
  write_functions(out, range(3, max_rank + 1))
  update_generated_source_code.update_generated_content(
      path=os.path.join(os.path.dirname(__file__), 'make_array.inc'),
      script_name=os.path.basename(__file__),
      new_content=out.getvalue(),
  )


if __name__ == '__main__':
  main()
