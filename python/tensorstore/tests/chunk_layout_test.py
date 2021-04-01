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
"""Tests for tensorstore.ChunkLayout."""

import numpy as np
import tensorstore as ts


def test_json():
  json_layout = {
      'inner_order': [1, 0],
      'grid_origin': [1, 2],
      'write_chunk': {
          'shape': [10, 11],
      },
  }
  assert json_layout == ts.ChunkLayout(json_layout).to_json()


def test_init():
  layout = ts.ChunkLayout(
      write_chunk=ts.ChunkLayout.Grid([10, 12]),
      read_chunk=ts.ChunkLayout.Grid([0, 6]),
      codec_chunk=ts.ChunkLayout.Grid([0, 3]),
      inner_order=[1, 0],
      grid_origin=[1, 2],
  )
  assert layout.ndim == 2
  np.testing.assert_array_equal(layout.grid_origin, [1, 2])
  np.testing.assert_array_equal(layout.write_chunk.shape, [10, 12])
  np.testing.assert_array_equal(layout.read_chunk.shape, [10, 6])
  np.testing.assert_array_equal(layout.codec_chunk.shape, [0, 3])
  assert layout.read_chunk_template == ts.IndexDomain(inclusive_min=[1, 2],
                                                      shape=[10, 6])
  assert layout.write_chunk_template == ts.IndexDomain(inclusive_min=[1, 2],
                                                       shape=[10, 12])
  assert layout.rank == 2
  np.testing.assert_array_equal(layout.inner_order, [1, 0])


def test_lifetime():
  shape = ts.ChunkLayout.Grid(shape=[1, 2, 3]).shape
  np.testing.assert_array_equal(shape, [1, 2, 3])


def test_tensorstore_layout():
  layout = ts.open(
      {
          'driver': 'zarr',
          'kvstore': {
              'driver': 'memory'
          },
          'metadata': {
              'chunks': [10, 11],
              'shape': [100, 200],
              'fill_value': None,
              'dtype': '<u2',
              'compressor': None,
              'filters': None,
              'order': 'C'
          },
      }, create=True).result().chunk_layout
  assert layout == ts.ChunkLayout(write_chunk=ts.ChunkLayout.Grid([10, 11]),
                                  inner_order=[0, 1])
