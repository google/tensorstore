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
"""Generates MatchesArray overloads for array_testutil.h."""

from __future__ import unicode_literals

import io
import os

import update_generated_source_code


def print_for_rank(rank, origin_type):
  """Print a single MatchesArray definition.

  Args:
    rank: Positive int.  Specifies the rank of the array.
    origin_type: One of `'span'` (to generate an overload where the offset is
      specified as a `span`), `'array`` (to generate an overload where the
      offset is specified as an array, in order to allow it to be called using a
      braced list), or `None` (to generate zero-origin overload).

  Returns:
    The overload definition as a `str`.
  """
  is_offset = (origin_type is not None)
  if origin_type == 'span':
    origin_parameter = ('span<const Index, {rank}> origin, '.format(rank=rank))
  elif origin_type == 'array':
    origin_parameter = ('const Index (&origin)[OriginRank], '.format(rank=rank))
  else:
    origin_parameter = ''
  origin_argument = 'origin, ' if is_offset else ''
  make_func = 'MakeOffsetArray' if is_offset else 'MakeArray'
  out = io.StringIO()
  origin_explanation = 'the specified origin' if is_offset else 'zero origin'
  out.write(r"""
/// Returns a GMock matcher that matches a rank-{rank} array with {origin_explanation}.
///
/// This overload can be called with a braced list.
///""".format(rank=rank, origin_explanation=origin_explanation))
  if is_offset:
    out.write(r"""
/// \param origin The expected origin vector of the array.""")
  out.write(r"""
/// \param element_matchers The matchers for each element of the array.
""")
  out.write('template <typename Element')
  for i in range(rank):
    out.write(', Index N%d' % i)
  if origin_type == 'array':
    out.write(', std::ptrdiff_t OriginRank')
  out.write('>\n')
  out.write('ArrayMatcher MatchesArray({origin_parameter}const '
            '::testing::Matcher<Element> (&element_matchers)'.format(
                origin_parameter=origin_parameter))
  for i in range(rank):
    out.write('[N%d]' % i)
  out.write(') {\n')
  if origin_type == 'array':
    out.write('static_assert(OriginRank == {rank}, '
              '"Origin vector must have length {rank}.");\n'.format(rank=rank))
  out.write(
      '  return '
      'MatchesArray<Element>({make_func}({origin_argument}element_matchers));'
      .format(make_func=make_func, origin_argument=origin_argument))
  out.write('\n}\n')
  return out.getvalue()


def write_functions(out, ranks):
  for origin_type in [None, 'span', 'array']:
    for rank in ranks:
      out.write(print_for_rank(rank, origin_type=origin_type))


def main():
  max_rank = 6

  # Write the rank-1 overloads to array_testutil.h, and write the higher-rank
  # overloads to array_testutil_matches_array.inc.
  out = io.StringIO()
  write_functions(out, [1])
  out.write("""

// Defines MatchesArray overloads for multi-dimensional arrays of rank 2 to {max_rank}.
#include "third_party/tensorstore/array_testutil_matches_array.inc"
""".format(max_rank=max_rank))

  update_generated_source_code.update_generated_content(
      path=os.path.join(os.path.dirname(__file__), 'array_testutil.h'),
      script_name=os.path.basename(__file__),
      new_content=out.getvalue(),
  )

  out = io.StringIO()
  write_functions(out, range(2, max_rank + 1))
  update_generated_source_code.update_generated_content(
      path=os.path.join(
          os.path.dirname(__file__), 'array_testutil_matches_array.inc'),
      script_name=os.path.basename(__file__),
      new_content=out.getvalue(),
  )


if __name__ == '__main__':
  main()
