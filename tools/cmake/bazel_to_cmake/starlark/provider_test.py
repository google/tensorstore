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

# pylint: disable=g-importing-member,invalid-name

import unittest

from .provider import provider


class TestProvider(unittest.TestCase):

  def test_generic_provider(self):
    X = provider(doc='foo')
    x = X(a=1, b=2)
    self.assertEqual(x.a, 1)
    self.assertEqual(x.b, 2)
    self.assertEqual('struct(a=1,b=2)', str(repr(x)))
    y = X(c=1)
    # Cannot assign
    with self.assertRaises(Exception) as context:
      x.a = 2

  def test_restricted_provider(self):
    Y = provider(doc='bar', fields=['a', 'b'])
    x = Y(a=1, b=2)
    self.assertEqual(x.a, 1)
    self.assertEqual(x.b, 2)
    self.assertEqual('struct(a=1,b=2)', str(repr(x)))
    Y(b=2)

    # Cannot assign
    with self.assertRaises(Exception) as _:
      x.a = 2

    # Invalid field.
    with self.assertRaises(Exception) as _:
      Y(c=2)
