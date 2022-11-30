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

# pylint: disable=g-importing-member

from .select import Select
import unittest

from .bazel_target import TargetId


class TestSelect(unittest.TestCase):

  def test_basic(self):
    left = Select({
        TargetId('@//conditions:default'): ['a'],
        TargetId('@foo//bar:baz'): ['b']
    })
    right = Select({
        TargetId('@//conditions:default'): ['c'],
        TargetId('@foo//bar:baz'): ['d']
    })
    added = left + ['x'] + right

    x = added.evaluate(lambda x: False)
    self.assertEqual(x, ['a', 'x', 'c'])

  def test_failure(self):
    cases = Select({
        TargetId('@foo//bar:baz'): ['b'],
        TargetId('@foo//bar:ball'): ['c'],
    })
    with self.assertRaises(Exception) as _:
      # Returns false for all cases, raising an error
      cases.evaluate(lambda x: False)
    with self.assertRaises(Exception) as _:
      # Returns true for all cases, raising an error
      cases.evaluate(lambda x: True)
