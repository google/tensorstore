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

from .struct import Struct


def test_struct():
  x = Struct(doc='foo', id=1)
  assert x.doc == 'foo'
  assert x.id == 1
  assert "struct(doc='foo',id=1)" == str(repr(x))
  with pytest.raises(Exception):
    x.doc = 'bar'


def test_struct_add():
  x = Struct(doc='foo', id=1)
  y = Struct(a=1, b='bar')
  assert x != y
  z = x + y
  assert z.a == 1
  with pytest.raises(Exception):
    z + x
