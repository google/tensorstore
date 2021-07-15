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
import pytest
import tensorstore as ts


def test_grid_json():
  json_grid = dict(
      shape=[5, None, 20],
      shape_soft_constraint=[None, 17, None],
      aspect_ratio=[1, 2, None],
      aspect_ratio_soft_constraint=[None, None, 3],
      elements=1000,
  )
  grid = ts.ChunkLayout.Grid(json_grid)
  assert grid.shape == (5, None, 20)
  assert grid.shape_soft_constraint == (None, 17, None)
  assert grid.aspect_ratio == (1, 2, None)
  assert grid.aspect_ratio_soft_constraint == (None, None, 3)
  assert grid.elements == 1000
  assert grid.elements_soft_constraint is None
  assert grid.to_json() == json_grid


def test_grid_update():
  grid = ts.ChunkLayout.Grid()
  assert grid.rank is None
  assert grid.ndim is None
  grid.update(shape=[2, 3, None])
  assert grid.rank == 3
  assert grid.shape == (2, 3, None)
  with pytest.raises(ValueError):
    grid.update(shape=[4, None, None])


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
      write_chunk=ts.ChunkLayout.Grid(shape=[10, 12]),
      read_chunk=ts.ChunkLayout.Grid(shape=[0, 6]),
      codec_chunk=ts.ChunkLayout.Grid(shape=[0, 3]),
      inner_order=[1, 0],
      grid_origin=[1, 2],
      finalize=True,
  )
  assert layout.ndim == 2
  assert layout.grid_origin == (1, 2)
  assert layout.write_chunk.shape == (10, 12)
  assert layout.read_chunk.shape == (10, 6)
  assert layout.codec_chunk.shape == (None, 3)
  assert layout.read_chunk_template == ts.IndexDomain(inclusive_min=[1, 2],
                                                      shape=[10, 6])
  assert layout.write_chunk_template == ts.IndexDomain(inclusive_min=[1, 2],
                                                       shape=[10, 12])
  assert layout.rank == 2
  np.testing.assert_array_equal(layout.inner_order, [1, 0])


def test_chunk_shape():
  layout = ts.ChunkLayout(chunk_shape=[10, None, None])
  assert layout.read_chunk.shape == (10, None, None)
  assert layout.write_chunk.shape == (10, None, None)


def test_chunk_aspect_ratio():
  layout = ts.ChunkLayout(chunk_aspect_ratio=[None, 2, 1])
  assert layout.read_chunk.aspect_ratio == (None, 2, 1)
  assert layout.write_chunk.aspect_ratio == (None, 2, 1)


def test_chunk_elements():
  layout = ts.ChunkLayout(chunk_elements=100000)
  assert layout.read_chunk.elements == 100000
  assert layout.write_chunk.elements == 100000


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
  assert layout == ts.ChunkLayout(
      grid_origin=[0, 0],
      chunk=ts.ChunkLayout.Grid(shape=[10, 11]),
      inner_order=[0, 1],
  )
