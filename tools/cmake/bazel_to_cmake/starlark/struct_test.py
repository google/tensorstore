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

from .struct import Struct
import unittest


class TestStruct(unittest.TestCase):

  def test_struct(self):
    x = Struct(doc='foo', id=1)
    self.assertEqual(x.doc, 'foo')
    self.assertEqual(x.id, 1)
    self.assertEqual("struct(doc='foo',id=1)", str(repr(x)))
    with self.assertRaises(Exception) as _:
      x.doc = 'bar'

  def test_struct_add(self):
    x = Struct(doc='foo', id=1)
    y = Struct(a=1, b='bar')
    self.assertNotEqual(x, y)
    z = x + y
    self.assertEqual(z.a, 1)
    with self.assertRaises(Exception) as _:
      z + x
