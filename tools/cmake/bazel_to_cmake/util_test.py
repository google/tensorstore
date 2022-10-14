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
"""Tests for utility functions."""

# pylint: disable=relative-beyond-top-level

from .util import cmake_is_true


def test_cmake_is_true():
  assert cmake_is_true("1")
  assert cmake_is_true("ON")
  assert cmake_is_true("YES")
  assert cmake_is_true("TRUE")
  assert cmake_is_true("Y")
  assert cmake_is_true("2")
  assert cmake_is_true("notfound")
  assert not cmake_is_true("0")
  assert not cmake_is_true("OFF")
  assert not cmake_is_true("NO")
  assert not cmake_is_true("FALSE")
  assert not cmake_is_true("N")
  assert not cmake_is_true("IGNORE")
  assert not cmake_is_true("NOTFOUND")
  assert not cmake_is_true("x-NOTFOUND")
  assert not cmake_is_true("")
  assert not cmake_is_true(None)
