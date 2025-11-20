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

import pickle
import threading
import time
from typing import Callable

import numpy as np
import pytest
import tensorstore as ts


def test_grid_json() -> None:
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


def test_grid_update() -> None:
  grid = ts.ChunkLayout.Grid()
  assert grid.rank is None
  assert grid.ndim is None
  grid.update(shape=[2, 3, None])
  assert grid.rank == 3
  assert grid.shape == (2, 3, None)
  with pytest.raises(ValueError):
    grid.update(shape=[4, None, None])


def test_pickle_grid() -> None:
  x = ts.ChunkLayout.Grid(shape=[10, 12])
  assert pickle.loads(pickle.dumps(x)) == x


def test_json() -> None:
  json_layout = {
      'inner_order': [1, 0],
      'grid_origin': [1, 2],
      'write_chunk': {
          'shape': [10, 11],
      },
  }
  assert json_layout == ts.ChunkLayout(json_layout).to_json()


def test_init() -> None:
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
  assert layout.read_chunk_template == ts.IndexDomain(
      inclusive_min=[1, 2], shape=[10, 6]
  )
  assert layout.write_chunk_template == ts.IndexDomain(
      inclusive_min=[1, 2], shape=[10, 12]
  )
  assert layout.rank == 2
  np.testing.assert_array_equal(layout.inner_order, [1, 0])


def test_chunk_shape() -> None:
  layout = ts.ChunkLayout(chunk_shape=[10, None, None])
  assert layout.read_chunk.shape == (10, None, None)
  assert layout.write_chunk.shape == (10, None, None)


def test_chunk_aspect_ratio() -> None:
  layout = ts.ChunkLayout(chunk_aspect_ratio=[None, 2, 1])
  assert layout.read_chunk.aspect_ratio == (None, 2, 1)
  assert layout.write_chunk.aspect_ratio == (None, 2, 1)


def test_chunk_elements() -> None:
  layout = ts.ChunkLayout(chunk_elements=100000)
  assert layout.read_chunk.elements == 100000
  assert layout.write_chunk.elements == 100000


def test_lifetime() -> None:
  shape = ts.ChunkLayout.Grid(shape=[1, 2, 3]).shape
  np.testing.assert_array_equal(shape, [1, 2, 3])


def test_tensorstore_layout() -> None:
  layout = (
      ts.open(
          {
              'driver': 'zarr',
              'kvstore': {'driver': 'memory'},
              'metadata': {
                  'chunks': [10, 11],
                  'shape': [100, 200],
                  'fill_value': None,
                  'dtype': '<u2',
                  'compressor': None,
                  'filters': None,
                  'order': 'C',
              },
          },
          create=True,
      )
      .result()
      .chunk_layout
  )
  assert layout == ts.ChunkLayout(
      grid_origin=[0, 0],
      chunk=ts.ChunkLayout.Grid(shape=[10, 11]),
      inner_order=[0, 1],
  )


def test_pickle_chunk_layout() -> None:
  x = ts.ChunkLayout(
      write_chunk=ts.ChunkLayout.Grid(shape=[10, 12]),
      read_chunk=ts.ChunkLayout.Grid(shape=[0, 6]),
      codec_chunk=ts.ChunkLayout.Grid(shape=[0, 3]),
      inner_order=[1, 0],
      grid_origin=[1, 2],
  )
  assert pickle.loads(pickle.dumps(x)) == x


def _run_threads(
    stop: threading.Event,
    read_props: Callable[[], None],
    update_props: Callable[[], None],
) -> None:
  threads = []
  for _ in range(4):
    threads.append(threading.Thread(target=read_props))
    threads.append(threading.Thread(target=update_props))

  for t in threads:
    t.start()

  time.sleep(0.3)
  stop.set()

  for t in threads:
    t.join()


def test_chunk_layout_concurrent() -> None:
  """Tests concurrent access to ChunkLayout properties."""
  layout = ts.ChunkLayout(chunk_shape=[10, 20], rank=2)
  stop = threading.Event()

  def read_props() -> None:
    while not stop.is_set():
      _ = layout.rank
      _ = layout.ndim
      _ = layout.inner_order
      _ = layout.write_chunk
      _ = layout.read_chunk
      _ = layout == ts.ChunkLayout(rank=2)
      _ = f'{layout}'
      _ = repr(layout)

  def update_props() -> None:
    time.sleep(0.01)
    i = 0
    while not stop.is_set():
      if (i % 2) == 0:
        layout.update(chunk_shape=[10, 20])
      else:
        layout.update(rank=2)
      i += 1

  _run_threads(stop, read_props, update_props)


def test_grid_concurrent() -> None:
  """Tests concurrent access to ChunkLayout.Grid properties."""
  grid = ts.ChunkLayout.Grid(shape=[10, 10])
  stop = threading.Event()

  def read_props() -> None:
    while not stop.is_set():
      _ = grid.rank
      _ = grid.ndim
      _ = grid.shape
      _ = grid.elements
      _ = grid == ts.ChunkLayout.Grid(rank=2)
      _ = f'{grid}'
      _ = repr(grid)

  def update_props() -> None:
    time.sleep(0.01)
    i = 0
    while not stop.is_set():
      if (i % 2) == 0:
        grid.update(shape=[10, 10])
      else:
        grid.update(rank=2, elements=100)
      i += 1

  _run_threads(stop, read_props, update_props)
