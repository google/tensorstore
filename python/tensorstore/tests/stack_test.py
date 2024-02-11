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
"""Tests for tensorstore.overlay."""

import numpy as np
import tensorstore as ts


def test_overlay():
  a = ts.array([1, 2, 3, 4], dtype=ts.int32)
  b = ts.array([5, 6, 7, 8], dtype=ts.int32)
  c = ts.overlay([a, b.translate_to[3]])
  np.testing.assert_equal(np.array([1, 2, 3, 5, 6, 7, 8], dtype=np.int32), c)


def test_stack():
  a = ts.array([1, 2, 3, 4], dtype=ts.int32)
  b = ts.array([5, 6, 7, 8], dtype=ts.int32)
  c = ts.stack([a, b])
  np.testing.assert_equal(
      np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32), c
  )


def test_concat():
  a = ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.int32)
  b = ts.array([[7, 8, 9], [10, 11, 12]], dtype=ts.int32)
  c = ts.concat([a, b], axis=0)
  np.testing.assert_equal(
      np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int32),
      c,
  )
