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
"""Tests for tensorstore.downsample."""

import numpy as np
import pytest
import tensorstore as ts

pytestmark = pytest.mark.asyncio


async def test_downsample_store_float32():
  t = ts.array(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))

  downsampled = ts.downsample(t, [1, 2], method='mean')

  np.testing.assert_equal(
      np.array([[1.5, 3], [4.5, 6]], dtype=np.float32), await
      downsampled.read())


async def test_downsample_store_uint32():
  t = ts.array(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32))

  downsampled = ts.downsample(t, [2, 2], method='mean')

  np.testing.assert_equal(
      np.array([[3, 4]], dtype=np.uint32), await
      downsampled.read())


async def test_downsample_spec():

  t = ts.array(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
  spec = t.spec()
  downsampled_spec = ts.downsample(spec, [1, 2], method='mean')

  downsampled = await ts.open(downsampled_spec)

  np.testing.assert_equal(
      np.array([[1.5, 3], [4.5, 6]], dtype=np.float32), await
      downsampled.read())
