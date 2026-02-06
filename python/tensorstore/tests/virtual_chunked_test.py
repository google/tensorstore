# Copyright 2021 The TensorStore Authors
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
"""Tests for tensorstore.virtual_chunked."""

import gc
from typing import Any, Callable
import weakref

import cloudpickle  # type: ignore[import-untyped]
import numpy as np
import tensorstore as ts

GcTester = Callable[[Any], None]


def test_read_pickle() -> None:
  """Tests reading, and that reading works after pickling."""

  def do_read(
      domain: ts.IndexDomain,
      array: np.ndarray,
      read_params: ts.VirtualChunkedReadParameters,
  ) -> None:
    del domain
    del read_params
    array[...] = 42

  t = ts.virtual_chunked(do_read, dtype=np.int32, shape=[2, 3])

  np.testing.assert_array_equal(
      t.read().result(), np.array([[42, 42, 42], [42, 42, 42]], dtype=np.int32)
  )

  t2 = cloudpickle.loads(cloudpickle.dumps(t))

  assert t2.schema == t.schema

  np.testing.assert_array_equal(
      t2.read().result(), np.array([[42, 42, 42], [42, 42, 42]], dtype=np.int32)
  )


def test_read_write_pickle() -> None:
  """Tests reading and writing, and that writing works after pickling."""
  array = np.zeros(shape=[4, 5], dtype=np.int32)

  def do_read(
      domain: ts.IndexDomain,
      chunk: np.ndarray,
      read_params: ts.VirtualChunkedReadParameters,
  ) -> None:
    del read_params
    chunk[...] = array[domain.index_exp]

  def do_write(
      domain: ts.IndexDomain,
      chunk: np.ndarray,
      write_params: ts.VirtualChunkedWriteParameters,
  ) -> None:
    del write_params
    array[domain.index_exp] = chunk

  t = ts.virtual_chunked(
      do_read,
      do_write,
      dtype=array.dtype,
      shape=array.shape,
      chunk_layout=ts.ChunkLayout(read_chunk_shape=(2, 3)),
  )

  t[1:3, 1:3] = 42

  np.testing.assert_array_equal(
      array,
      [
          [0, 0, 0, 0, 0],
          [0, 42, 42, 0, 0],
          [0, 42, 42, 0, 0],
          [0, 0, 0, 0, 0],
      ],
  )

  np.testing.assert_array_equal(
      t,
      [
          [0, 0, 0, 0, 0],
          [0, 42, 42, 0, 0],
          [0, 42, 42, 0, 0],
          [0, 0, 0, 0, 0],
      ],
  )

  array2, t2 = cloudpickle.loads(cloudpickle.dumps((array, t)))

  t2[1:3, 1:3] = 43

  np.testing.assert_array_equal(
      array2,
      [
          [0, 0, 0, 0, 0],
          [0, 43, 43, 0, 0],
          [0, 43, 43, 0, 0],
          [0, 0, 0, 0, 0],
      ],
  )

  np.testing.assert_array_equal(
      t2,
      [
          [0, 0, 0, 0, 0],
          [0, 43, 43, 0, 0],
          [0, 43, 43, 0, 0],
          [0, 0, 0, 0, 0],
      ],
  )

  np.testing.assert_array_equal(
      array,
      [
          [0, 0, 0, 0, 0],
          [0, 42, 42, 0, 0],
          [0, 42, 42, 0, 0],
          [0, 0, 0, 0, 0],
      ],
  )

  np.testing.assert_array_equal(
      t,
      [
          [0, 0, 0, 0, 0],
          [0, 42, 42, 0, 0],
          [0, 42, 42, 0, 0],
          [0, 0, 0, 0, 0],
      ],
  )


async def test_read_adapt() -> None:
  """Tests reading where the `read_function` itself reads from a TensorStore."""
  a = ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.int32)

  async def do_read(
      domain: ts.IndexDomain,
      array: np.ndarray,
      read_params: ts.VirtualChunkedReadParameters,
  ) -> None:
    del read_params
    array[...] = (await a[domain].read()) + 100

  t = ts.virtual_chunked(do_read, dtype=a.dtype, domain=a.domain)

  np.testing.assert_array_equal(
      await t.read(),
      np.array([[101, 102, 103], [104, 105, 106]], dtype=np.int32),
  )


async def test_string_read() -> None:
  """Tests that reading works with a dtype of ustring.

  This relies on an additional copy since TensorStore does not share the same
  string representation as Python.
  """

  async def do_read(
      domain: ts.IndexDomain,
      array: np.ndarray,
      read_params: ts.VirtualChunkedReadParameters,
  ) -> None:
    del read_params
    for i, x in enumerate(domain[0]):
      for j, y in enumerate(domain[1]):
        array[i, j] = '%d,%d' % (x, y)

  t = ts.virtual_chunked(
      do_read,
      dtype=ts.ustring,
      shape=[3, 4],
      chunk_layout=ts.ChunkLayout(read_chunk_shape=[2, 3]),
  )
  np.testing.assert_array_equal(
      await t.read(),
      np.array(
          [
              ['0,0', '0,1', '0,2', '0,3'],
              ['1,0', '1,1', '1,2', '1,3'],
              ['2,0', '2,1', '2,2', '2,3'],
          ],
          dtype=object,
      ),
  )


async def test_string_write() -> None:
  """Tests that writing works with a dtype of ustring.

  This relies on an additional copy since TensorStore does not share the same
  string representation as Python.
  """

  arr = np.zeros(shape=(2,), dtype=object)

  async def do_write(
      domain: ts.IndexDomain,
      array: np.ndarray,
      write_params: ts.VirtualChunkedWriteParameters,
  ) -> None:
    del write_params
    arr[domain.index_exp] = array

  t = ts.virtual_chunked(
      write_function=do_write, dtype=ts.ustring, shape=arr.shape
  )
  await t.write(['a', 'b'])
  np.testing.assert_array_equal(arr, ['a', 'b'])


def test_gc(gc_tester: GcTester) -> None:
  """Tests that a cyclic reference involving the TensorStore is collected."""

  def do_read(
      domain: ts.IndexDomain,
      array: np.ndarray,
      read_params: ts.VirtualChunkedReadParameters,
  ) -> None:
    del domain
    del read_params
    array[...] = 42

  t = ts.virtual_chunked(do_read, dtype=np.int32, shape=[2, 3])

  do_read.t = t  # type: ignore

  gc_tester(do_read)
  gc_tester(t)


def test_spec_gc(gc_tester: GcTester) -> None:
  """Tests that a cyclic reference involving the Spec is collected."""

  def do_read(
      domain: ts.IndexDomain,
      array: np.ndarray,
      read_params: ts.VirtualChunkedReadParameters,
  ) -> None:
    del domain
    del read_params
    array[...] = 42

  t = ts.virtual_chunked(do_read, dtype=np.int32, shape=[2, 3])
  spec = t.spec()
  do_read.spec = spec  # type: ignore
  gc_tester(spec)


def test_spec_keep_alive() -> None:
  def do_read(
      domain: ts.IndexDomain,
      array: np.ndarray,
      read_params: ts.VirtualChunkedReadParameters,
  ) -> None:
    del domain
    del read_params
    array[...] = 42

  t = ts.virtual_chunked(do_read, dtype=np.int32, shape=[2, 3])
  ref = weakref.ref(do_read)
  del do_read
  spec = t.spec()
  del t

  gc.collect()

  # Verify that `spec` holds a strong reference to `do_read`.

  assert ref() is not None
  del spec


def test_gc_future(gc_tester: GcTester) -> None:
  """Tests that a cyclic reference involving a Future is collected."""

  def do_read(
      domain: ts.IndexDomain,
      array: np.ndarray,
      read_params: ts.VirtualChunkedReadParameters,
  ) -> ts.Future:
    del domain
    del array
    del read_params
    promise, future = ts.Promise.new()
    do_read.promise = promise  # type: ignore
    return future

  t = ts.virtual_chunked(do_read, dtype=np.int32, shape=[2, 3])

  future = t.read()
  do_read.future = future  # type: ignore

  gc_tester(t)


def test_read_batch() -> None:
  """Tests reading both with and without a batch."""
  array = ts.array(np.arange(np.prod(shape := (4, 5))).reshape(shape))
  # Avoid sharing the common threadpool with array
  context = ts.Context({'data_copy_concurrency': {'limit': 1}})

  def do_batched_read(
      domain: ts.IndexDomain,
      chunk: np.ndarray,
      read_params: ts.VirtualChunkedReadParameters,
  ) -> None:
    assert isinstance(read_params.batch, ts.Batch)
    chunk[...] = array[domain].read(batch=read_params.batch).result()

  t = ts.virtual_chunked(
      do_batched_read,
      None,
      dtype=array.dtype,
      shape=array.shape,
      chunk_layout=ts.ChunkLayout(read_chunk_shape=(2, 3)),
      context=context,
  )

  with ts.Batch() as b:
    f = t[1:3, 1:3].read(batch=b)

  np.testing.assert_array_equal(f.result(), array[1:3, 1:3])

  def do_unbatched_read(
      domain: ts.IndexDomain,
      chunk: np.ndarray,
      read_params: ts.VirtualChunkedReadParameters,
  ) -> None:
    assert read_params.batch is None
    chunk[...] = array[domain].read(batch=read_params.batch).result()

  t2 = ts.virtual_chunked(
      do_unbatched_read,
      None,
      dtype=array.dtype,
      shape=array.shape,
      chunk_layout=ts.ChunkLayout(read_chunk_shape=(2, 3)),
      context=context,
  )

  np.testing.assert_array_equal(t2[1:3, 1:3].read().result(), array[1:3, 1:3])
