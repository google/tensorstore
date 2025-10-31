# Copyright 2025 The TensorStore Authors
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
"""Tests for tensorstore.Batch."""

import numpy as np
import pytest
import tensorstore as ts


async def _get_store() -> ts.TensorStore:
  store = await ts.open(
      {
          'driver': 'zarr',
          'kvstore': 'memory://',
      },
      dtype=ts.uint32,
      shape=[10, 20],
      chunk_layout=ts.ChunkLayout(read_chunk_shape=[5, 10]),
      create=True,
      delete_existing=True,
  )
  await store.write(np.arange(200, dtype=np.uint32).reshape((10, 20)))
  return store


async def test_batch_submit():
  store = await _get_store()
  batch = ts.Batch()
  f1 = store[:5, :10].read(batch=batch)
  f2 = store[5:, 10:].read(batch=batch)
  batch.submit()
  a1 = await f1
  a2 = await f2
  np.testing.assert_array_equal(
      np.arange(200, dtype=np.uint32).reshape((10, 20))[:5, :10], a1
  )
  np.testing.assert_array_equal(
      np.arange(200, dtype=np.uint32).reshape((10, 20))[5:, 10:], a2
  )


async def test_batch_del():
  store = await _get_store()
  batch = ts.Batch()
  f1 = store[:5, :10].read(batch=batch)
  f2 = store[5:, 10:].read(batch=batch)
  del batch
  a1 = await f1
  a2 = await f2
  np.testing.assert_array_equal(
      np.arange(200, dtype=np.uint32).reshape((10, 20))[:5, :10], a1
  )
  np.testing.assert_array_equal(
      np.arange(200, dtype=np.uint32).reshape((10, 20))[5:, 10:], a2
  )


async def test_batch_submitted_error():
  store = await _get_store()
  batch = ts.Batch()
  batch.submit()
  with pytest.raises(ValueError, match='batch was already submitted'):
    store[:5, :10].read(batch=batch)
