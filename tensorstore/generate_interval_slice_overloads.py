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
"""Generates the DimExpression::*Interval methods in dim_expression.h."""

# pylint: disable=line-too-long

from __future__ import unicode_literals

import io
import json
import os

import update_generated_source_code


def main():
  out = io.StringIO()
  out.write('  // The following code is automatically generated.  '
            'Do not modify directly.\n')

  for translate in [False, True]:
    for orig_prefix, interval_form, stop_name in [('Closed', 'closed', 'stop'),
                                                  ('HalfOpen', 'half_open',
                                                   'stop'),
                                                  ('Sized', 'sized', 'size')]:
      translate_prefix = 'Translate' if translate else ''
      prefix = translate_prefix + orig_prefix
      stop_type_name = stop_name.title()
      for i in range(8):
        template_params = []
        function_params = []
        stored_types = []
        for param_i, (
            param_name,
            param_type_name,
            param_type_default,
            param_default,
        ) in enumerate([
            ('start', 'Start', '', ''),
            (stop_name, stop_type_name, '', ''),
            ('strides', 'Strides', ' = Index', ' = 1'),
        ]):
          is_braced = (i >> param_i) & 1
          if not is_braced:
            template_params.append('typename %s%s' %
                                   (param_type_name, param_type_default))
            function_params.append('const %s& %s%s' %
                                   (param_type_name, param_name, param_default))
            stored_types.append(param_type_name)
          else:
            function_params.append('const Index (&%s)[Rank]' % param_name)
            stored_types.append('const Index (&)[Rank]')
        if i != 0:
          template_params.append('DimensionIndex Rank')
        out.write('\n')
        if i != 0:
          out.write("""
// Overload that permits arguments to be specified as braced lists.
""")
        else:
          doc_comment = (
              """  /// Extracts a %s interval from the selected dimensions with optional
  /// striding.
  ///
  /// The domain of each selected dimension is transformed by
  /// Extract%sStridedSlice using the corresponding components of the
  /// `start`, `%s`, and `strides` vectors.  In the simple case that the stride
  /// component is `1`, the new domain is simply restricted to the specified
  /// interval, with the new origin equal to the specified `start` component.
  /// In the general case with a stide component not equal to `1`, the new
  /// origin is equal to the `start` component divided by the `strides`
  /// component, rounded towards zero; in this case, the `Translate%sInterval`
  /// operation, which ensures an origin of 0, may be more convenient.
  ///
  /// The new dimension selection is the same as the prior dimension selection,
  /// with a static rank equal to the merged static rank of the prior dimension
  /// selection and the static extents of the `start`, `%s`, and `strides`
  /// vectors.
  ///
""" % (interval_form.replace(
                  '_', '-'), orig_prefix, stop_name, orig_prefix, stop_name))
          if interval_form == 'closed':
            doc_comment += (
                """  /// For example: `Dims(0, 2).ClosedInterval({1, 8}, {4, 3}, {1, -2})` has the
  /// following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[0, 6], [2, 5], [0, 9]``
  ///      - ``[1, 4], [2, 5], [-4, -2]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"x", "y", "z"}``
  ///    * - Equivalent input indices
  ///      - ``{2, 3, 6}``
  ///      - ``{2, 3, -3}``
  ///    * - Equivalent input indices
  ///      - ``{x, y, z * -2}``
  ///      - ``{x, y, z}``
  ///
  /// where ``x`` is any index in ``[1, 4]``, ``y`` is any index in
  /// ``[2, 5]``, and ``z`` is any index in ``[-4, -2]``.
  ///
  /// Note that in the case of a stride component not equal to `1` or `-1`, if
  /// the `start` component is not evenly divisible by the stride, the
  /// transformation involves an additional offset.
  ///
  /// For example: `Dims(0, 2).ClosedInterval({1, 9}, {4, 3}, {1, -2})` has the
  /// following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[0, 6], [2, 5], [0, 9]``
  ///      - ``[1, 4], [2, 5], [-4, -1]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"x", "y", "z"}``
  ///    * - Equivalent input indices
  ///      - ``{2, 3, 7}``
  ///      - ``{2, 3, -3}``
  ///    * - Equivalent input indices
  ///      - ``{x, y, z * -2 + 1}``
  ///      - ``{x, y, z}``
  ///
  /// where ``x`` is any index in ``[1, 4]``, ``y`` is any index in
  /// ``[2, 5]``, and ``z`` is any index in ``[-4, -1]``.
  ///
""")
          elif interval_form == 'half_open':
            doc_comment += (
                """  /// For example: `Dims(0, 2).HalfOpenInterval({1, 8}, {4, 3}, {1, -2})` has the
  /// following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[0, 6], [2, 5], [0, 9]``
  ///      - ``[1, 3], [2, 5], [-4, -2]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"x", "y", "z"}``
  ///    * - Equivalent input indices
  ///      - ``{2, 3, 6}``
  ///      - ``{2, 3, -3}``
  ///    * - Equivalent input indices
  ///      - ``{x, y, z * -2}``
  ///      - ``{x, y, z}``
  ///
  /// where ``x`` is any index in ``[1, 4]``, ``y`` is any index in
  /// ``[2, 5]``, and ``z`` is any index in ``[-4, -2]``.
  ///
""")
          elif interval_form == 'sized':
            doc_comment += (
                """  /// For example: `Dims(0, 2).SizedInterval({1, 8}, {3, 2}, {1, -2})` has the
  /// following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[0, 6], [2, 5], [0, 9]``
  ///      - ``[1, 3], [2, 5], [-4, -3]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"x", "y", "z"}``
  ///    * - Equivalent input indices
  ///      - ``{2, 3, 6}``
  ///      - ``{2, 3, -3}``
  ///    * - Equivalent input indices
  ///      - ``{x, y, z * -2}``
  ///      - ``{x, y, z}``
  ///
  /// where ``x`` is any index in ``[1, 3]``, ``y`` is any index in
  /// ``[2, 5]``, and ``z`` is any index in ``[-4, -3]``.
  ///
""")

          doc_comment += (
              r"""/// \requires `Start`, `%s`, and `Strides` satisfy the IsIndexVectorOrScalar
  ///     concept with static extents compatible with each other and with the
  ///     static rank of the dimension selection.
  /// \param start The index vector specifying the start indices for each
  ///     selected dimension.  May be a braced list, e.g. ``{1, 2, 3}``.
  ///     May also be a scalar, e.g. `5`, in which case the same start index is
  ///     used for all selected dimensions.
""" % (stop_type_name))
          if interval_form == 'sized':
            doc_comment += (
                r"""  /// \param size The size vector specifying the size of the domain for each
  ///     selected dimension.  May be a braced list or scalar.
""")
          else:
            doc_comment += (
                r"""  /// \param stop The index vector specifying the stop indices for each selected
  ///     dimension.  May be a braced list or scalar.
""")
          doc_comment += (
              r"""  /// \param strides The index vector specifying the stride value for each
  ///     selected dimension.  May be a braced list or scalar.  If not
  ///     specified, defaults to 1.
  /// \error `absl::StatusCode::kInvalidArgument` if the extents of the `start`,
  ///     `{stop}`, or `strides` vectors do not match the number of selected
  ///     dimensions.
  /// \error `absl::StatusCode::kInvalidArgument` or
  ///     `absl::StatusCode::kOutOfRange` if the `start`, `{stop}`, and
  ///     `strides` values are invalid or specify a slice outside the effective
  ///     bounds for a given dimension (implicit lower/upper bounds are treated
  ///     as -/+inf).
  /// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
  ///     when computing the resultant transform.
""".format(stop=stop_name))
          if not translate:
            out.write(doc_comment)
          else:
            out.write('/// Equivalent to '
                      '`%sInterval(start, %s, strides).TranslateTo(0)`.\n' %
                      (orig_prefix, stop_name))
        out.write('  template <%s>\n' % ', '.join(template_params))
        out.write('  IntervalSliceOpExpr<%s>' % (', '.join(stored_types)))
        out.write('  %sInterval(%s) const {\n' %
                  (prefix, ', '.join(function_params)))
        out.write(
            '    return {{IntervalForm::%s, %s, start, %s, strides}, *this};\n'
            % (interval_form, json.dumps(translate), stop_name))
        out.write('  }\n')

  update_generated_source_code.update_generated_content(
      path=os.path.join(
          os.path.dirname(__file__), 'index_space', 'dim_expression.h'),
      script_name=os.path.basename(__file__),
      new_content=out.getvalue(),
  )


if __name__ == '__main__':
  main()
