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

import pytest

from .bazel_target import TargetId
from .select import Select, SelectExpression  # pylint: disable=multiple-import


def test_basic():
  left = Select({
      TargetId.parse('@//conditions:default'): ['a'],
      TargetId.parse('@foo//bar:baz'): ['b'],
  })
  right = Select({
      TargetId.parse('@//conditions:default'): ['c'],
      TargetId.parse('@foo//bar:baz'): ['d'],
  })
  added = left + ['x'] + right

  assert isinstance(added, SelectExpression)
  x = added.evaluate(lambda x: False)
  assert x == ['a', 'x', 'c']


def test_or():
  left = Select({
      TargetId.parse('@//conditions:default'): {'a': 'a_value'},
      TargetId.parse('@foo//bar:baz'): {'b': 'b_value'},
  })
  right = Select({
      TargetId.parse('@//conditions:default'): {'c': 'c_value'},
      TargetId.parse('@foo//bar:baz'): {'d': 'd_value'},
  })
  added = left | {'e': 'e_value'} | right

  assert isinstance(added, SelectExpression)
  x = added.evaluate(lambda x: False)
  assert x == {'a': 'a_value', 'e': 'e_value', 'c': 'c_value'}


def test_failure():
  cases = Select({
      TargetId.parse('@foo//bar:baz'): ['b'],
      TargetId.parse('@foo//bar:ball'): ['c'],
  })
  with pytest.raises(Exception):
    # Returns false for all cases, raising an error
    cases.evaluate(lambda x: False)
  with pytest.raises(Exception):
    # Returns true for all cases, raising an error
    cases.evaluate(lambda x: True)
