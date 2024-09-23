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
"""Tests for tensorstore.TensorStore."""

import copy
import pickle
import re
import tempfile
import time

import numpy as np
import pytest
import tensorstore as ts

pytestmark = pytest.mark.asyncio


async def test_open_array_driver():
  t = await ts.open({
      "driver": "array",
      "array": [[1, 2, 3], [4, 5, 6]],
      "dtype": "int32",
  })
  assert t.domain == ts.IndexDomain(shape=[2, 3])
  assert t.rank == 2
  assert t.ndim == 2
  assert t.origin == (0, 0)
  assert t.shape == (2, 3)
  assert t.dtype == ts.int32
  assert t.readable == True
  assert t.writable == True
  assert t.mode == "rw"
  a = np.array(t)
  assert a.dtype == np.int32
  np.testing.assert_equal(a, [[1, 2, 3], [4, 5, 6]])

  t[1, 1] = np.int32(7)
  np.testing.assert_equal(np.array(t), [[1, 2, 3], [4, 7, 6]])

  t[1, 1] = 8
  np.testing.assert_equal(np.array(t), [[1, 2, 3], [4, 8, 6]])

  assert (await t.read()).flags.carray
  assert (await t.read(order="C")).flags.carray
  assert (await t.read(order="F")).flags.fortran

  with pytest.raises(
      TypeError, match=re.escape("`order` must be specified as 'C' or 'F'")
  ):
    await t.read(order="X")


async def test_resize():
  arr = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  t = await ts.open(
      {
          "driver": "zarr",
          "kvstore": "memory://",
          "metadata": {
              "shape": arr.shape,
              "chunks": (1, 1),
              "dtype": arr.dtype.str,
          },
      },
      open=True,
      create=True,
  )
  await t.write(arr)
  await t.resize(exclusive_max=(3, 2))
  a = await t.read()
  np.testing.assert_equal(a, [[1, 2], [4, 5], [7, 8]])
  await t.resize()  # Resize with no arguments does nothing.
  a = await t.read()
  np.testing.assert_equal(a, [[1, 2], [4, 5], [7, 8]])


async def test_array():
  t = ts.array(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64))
  assert t.spec().to_json(include_defaults=False) == {
      "driver": "array",
      "array": [[1, 2, 3], [4, 5, 6]],
      "dtype": "int64",
      "transform": {
          "input_inclusive_min": [0, 0],
          "input_exclusive_max": [2, 3],
      },
  }

  assert t.T.spec().to_json(include_defaults=False) == {
      "driver": "array",
      "array": [[1, 4], [2, 5], [3, 6]],
      "dtype": "int64",
      "transform": {
          "input_inclusive_min": [0, 0],
          "input_exclusive_max": [3, 2],
      },
  }

  assert t[0].spec().to_json(include_defaults=False) == {
      "driver": "array",
      "array": [1, 2, 3],
      "dtype": "int64",
      "transform": {
          "input_inclusive_min": [0],
          "input_exclusive_max": [3],
      },
  }


async def test_array_no_copy():
  with pytest.raises(
      ValueError, match=r"Unable to avoid copy while creating an array"
  ):
    ts.array([1, 2, 3], copy=False)

  x = np.array([1, 2, 3], dtype=np.int32)

  y = ts.array(x, copy=False)

  x[0] = 4
  assert (await y[0].read()) == 4

  y = ts.array(x, copy=True)
  x[0] = 5
  assert (await y[0].read()) == 4


async def test_array_no_write():
  x = np.array([1, 2, 3], dtype=np.int32)

  y = ts.array(x)
  assert y.writable

  y = ts.array(x, write=False)
  assert not y.writable

  x.setflags(write=False)
  y = ts.array(x)
  assert not y.writable

  y = ts.array(x, write=True)
  x.setflags(write=True)
  x[0] = 4
  assert (await y[0].read()) == 1
  x.setflags(write=False)

  with pytest.raises(
      ValueError,
      match=r"Unable to avoid copy while creating an array",
  ):
    ts.array(x, write=True, copy=False)


async def test_open_ustring_dtype():
  t = await ts.open({
      "driver": "array",
      "array": ["this", "is", "a", "string", "array"],
      "dtype": "ustring",
  })
  assert t.domain == ts.IndexDomain(shape=[5])
  assert t.dtype == ts.ustring
  a = await t.read()
  assert a.dtype == object
  np.testing.assert_equal(
      a, np.array(["this", "is", "a", "string", "array"], dtype=object)
  )


async def test_cast():
  t = ts.array(np.array([0, 1, 2, 3], dtype=np.int64))
  t_string = t.astype(bytes)
  np.testing.assert_equal(await t_string.read(), [b"0", b"1", b"2", b"3"])
  t_bool = ts.cast(t, bool)
  np.testing.assert_equal(await t_bool.read(), [False, True, True, True])


async def test_local_n5():
  with tempfile.TemporaryDirectory() as dir_path:
    dataset = ts.open({
        "driver": "n5",
        "kvstore": {
            "driver": "file",
            "path": dir_path,
        },
        "metadata": {
            "compression": {"type": "gzip"},
            "dataType": "uint32",
            "dimensions": [1000, 20000],
            "blockSize": [10, 10],
        },
        "create": True,
        "delete_existing": True,
    }).result()
    dataset[80:82, 99:102] = [[1, 2, 3], [4, 5, 6]]
    np.testing.assert_equal(
        [[1, 2, 3], [4, 5, 6], [0, 0, 0]],
        dataset[80:83, 99:102].read().result(),
    )


async def test_memory_n5_cache_open():
  ts.open({
      "context": {"cache_pool": {"total_bytes_limit": 1000000}},
      "driver": "n5",
      "kvstore": {
          "driver": "memory",
      },
      "metadata": {
          "compression": {"type": "gzip"},
          "dataType": "uint32",
          "dimensions": [1000, 20000],
          "blockSize": [10, 10],
      },
      "create": True,
      "delete_existing": True,
  }).result()


async def test_open_error_message():
  with pytest.raises(
      ValueError, match='.*Error parsing object member "driver": .*'
  ):
    await ts.open({"invalid": "key"})

  with pytest.raises(ValueError, match="Expected object, but received: 3"):
    await ts.open(3)


async def test_pickle():
  with tempfile.TemporaryDirectory() as dir_path:
    context = ts.Context({"cache_pool": {"total_bytes_limit": 1000000}})
    spec = {
        "driver": "n5",
        "kvstore": {
            "driver": "file",
            "path": dir_path,
        },
        "metadata": {
            "compression": {
                "type": "raw",
            },
            "dataType": "uint32",
            "dimensions": [100, 100],
            "blockSize": [10, 10],
        },
        "recheck_cached_data": False,
        "recheck_cached_metadata": False,
        "create": True,
        "open": True,
    }
    t1 = await ts.open(spec, context=context)
    t2 = await ts.open(spec, context=context)

    pickled = pickle.dumps([t1, t2])
    unpickled = pickle.loads(pickled)
    new_t1, new_t2 = unpickled

    assert new_t1[0, 0].read().result() == 0
    assert new_t2[0, 0].read().result() == 0
    new_t1[0, 0] = 42

    # Delete data
    await ts.open(spec, create=True, delete_existing=True)

    # new_t1 still sees old data in cache
    assert new_t1[0, 0].read().result() == 42

    # new_t2 shares cache with new_t1
    assert new_t2[0, 0].read().result() == 42


async def test_pickle_read_write_mode():
  with tempfile.TemporaryDirectory() as dir_path:
    spec = {
        "driver": "n5",
        "kvstore": {
            "driver": "file",
            "path": dir_path,
        },
        "metadata": {
            "compression": {
                "type": "raw",
            },
            "dataType": "uint32",
            "dimensions": [100, 100],
            "blockSize": [10, 10],
        },
        "recheck_cached_data": False,
        "recheck_cached_metadata": False,
        "create": True,
        "open": True,
    }
    t1 = await ts.open(spec, write=True)

    assert not t1.readable
    assert t1.writable

    pickled = pickle.dumps(t1)
    new_t1 = pickle.loads(pickled)

    assert not new_t1.readable
    assert new_t1.writable


async def test_copy():
  with tempfile.TemporaryDirectory() as dir_path:
    context = ts.Context({"cache_pool": {"total_bytes_limit": 1000000}})
    spec = {
        "driver": "zarr",
        "kvstore": {
            "driver": "file",
            "path": dir_path,
        },
        "recheck_cached_data": False,
        "recheck_cached_metadata": False,
        "create": True,
        "open": True,
    }
    t1 = await ts.open(spec, dtype=ts.uint16, shape=[100, 100], context=context)

    t2 = copy.copy(t1)
    assert t1 is t2

    t3 = copy.deepcopy(t1)

    t1[0, 0] = 42

    assert t1[0, 0].read().result() == 42

    # t3 can read data written with t1
    assert t3[0, 0].read().result() == 42

    t1[0, 0] = 43
    assert t1[0, 0].read().result() == 43

    # t3 sees stale data because it doesn't share a cache with t1
    assert t3[0, 0].read().result() == 42


async def test_write_json():
  t = await ts.open(
      {"driver": "array", "dtype": "json", "array": [1, {"a": 2}, 3], "rank": 1}
  )
  assert await t[1].read() == {"a": 2}
  assert await t[0].read() == 1
  assert await t[2].read() == 3
  np.testing.assert_equal(
      await t.read(), np.array([1, {"a": 2}, 3], dtype=object)
  )
  t[0] = {"x": 3}
  np.testing.assert_equal(
      await t.read(), np.array([{"x": 3}, {"a": 2}, 3], dtype=object)
  )
  with pytest.raises(TypeError):
    t[1] = object()
  np.testing.assert_equal(
      await t.read(), np.array([{"x": 3}, {"a": 2}, 3], dtype=object)
  )
  t[2] = [1, 2, 3]
  np.testing.assert_equal(
      await t.read(), np.array([{"x": 3}, {"a": 2}, [1, 2, 3]], dtype=object)
  )
  await t.write([1, 2, "abc"])
  np.testing.assert_equal(await t.read(), np.array([1, 2, "abc"], dtype=object))


async def test_write_ustring():
  t = await ts.open({
      "driver": "array",
      "dtype": "ustring",
      "array": ["abc", "x", "y"],
      "rank": 1,
  })
  assert await t[0].read() == "abc"
  np.testing.assert_equal(
      await t.read(), np.array(["abc", "x", "y"], dtype=object)
  )
  t[0] = "foo"
  np.testing.assert_equal(
      await t.read(), np.array(["foo", "x", "y"], dtype=object)
  )
  with pytest.raises(TypeError):
    t[1] = 3
  with pytest.raises(TypeError):
    t[1] = b"foo"


async def test_write_string():
  t = await ts.open({
      "driver": "array",
      "dtype": "string",
      "array": ["abc", "x", "y"],
      "rank": 1,
  })
  assert await t[0].read() == b"abc"
  np.testing.assert_equal(
      await t.read(), np.array([b"abc", b"x", b"y"], dtype=object)
  )
  t[0] = b"foo"
  np.testing.assert_equal(
      await t.read(), np.array([b"foo", b"x", b"y"], dtype=object)
  )
  with pytest.raises(TypeError):
    t[1] = "abc"


async def test_instantiation():
  with pytest.raises(TypeError):
    ts.TensorStore()


async def test_assume_metadata():
  t = await ts.open(
      {
          "driver": "zarr",
          "kvstore": "memory://",
      },
      dtype=ts.uint32,
      shape=[2, 3, 4],
      open=True,
      assume_metadata=True,
  )
  assert await t.kvstore.list() == []


async def test_storage_statistics():
  t = await ts.open(
      {
          "driver": "zarr",
          "kvstore": "memory://",
      },
      dtype=ts.uint8,
      shape=[100, 200, 300],
      chunk_layout=ts.ChunkLayout(read_chunk_shape=[10, 20, 30]),
      create=True,
  )

  transformed = t[(1, 1, 1):(20, 5, 5)]
  assert await transformed.storage_statistics(
      query_not_stored=True
  ) == ts.TensorStore.StorageStatistics(not_stored=True)
  await transformed.write(42)
  assert await transformed.storage_statistics(
      query_not_stored=True
  ) == ts.TensorStore.StorageStatistics(not_stored=False)
  assert await transformed.storage_statistics(
      query_not_stored=True, query_fully_stored=True
  ) == ts.TensorStore.StorageStatistics(not_stored=False, fully_stored=True)


async def test_storage_statistics_pickle():
  x = ts.TensorStore.StorageStatistics(not_stored=True, fully_stored=False)
  assert pickle.loads(pickle.dumps(x)) == x


async def test_tensorstore_ocdbt_zarr_repr():
  arr = ts.open(
      {
          "driver": "zarr",
          "kvstore": {
              "driver": "ocdbt",
              "base": "memory://",
              "path": "my_array/",
          },
      },
      shape=[1000, 2000, 3000],
      dtype=ts.float32,
      open=True,
      create=True,
  ).result()
  repr(arr)


async def test_spec_open_mode():
  spec = ts.Spec({
      "driver": "zarr",
      "kvstore": "memory://",
      "schema": {"dtype": "uint32", "domain": {"shape": [100, 200]}},
  })

  for open_mode_kwargs in [
      {"create": True},
      {"delete_existing": True, "create": True},
      {"open": True},
      {"open": True, "create": True},
      {"open": True, "assume_metadata": True},
      {"open": True, "assume_cached_metadata": True},
  ]:
    spec_copy = spec.copy()
    open_mode = ts.OpenMode(**open_mode_kwargs)
    spec_copy.update(**open_mode_kwargs)
    assert spec_copy.open_mode == open_mode

    spec_copy = spec.copy()
    spec_copy.update(open_mode=open_mode)
    assert spec_copy.open_mode == open_mode

    context = None
    if open_mode == ts.OpenMode(open=True):
      context = ts.Context()
      await ts.open(spec, create=True, context=context)
    store = await ts.open(spec, context=context, **open_mode_kwargs)

    requested_spec = store.spec(**open_mode_kwargs)
    assert requested_spec.open_mode == open_mode

    store = await ts.open(spec, context=context, open_mode=open_mode)
    requested_spec = store.spec(**open_mode_kwargs)
    assert requested_spec.open_mode == open_mode


@pytest.mark.parametrize("writable", [True, False])
async def test_zero_copy(writable: bool):
  store = await ts.open(
      {"driver": "zarr3", "kvstore": "memory://"},
      dtype=ts.uint32,
      shape=[64],
      create=True,
  )
  arr = np.full(shape=[64], fill_value=42, dtype=np.uint32)
  arr_maybe_immutable = arr[...]
  assert arr_maybe_immutable is not arr
  arr_maybe_immutable.setflags(write=writable)
  await store.write(
      arr_maybe_immutable, can_reference_source_data_indefinitely=True
  )
  np.testing.assert_equal(42, await store.read())
  # Modify arr.  This violates the guarantee indicated by
  # `can_reference_source_data_indefinitely=True` but is done here for testing
  # purposes.
  arr[...] = 43
  np.testing.assert_equal(43, await store.read())


def test_issue_168():
  t = ts.array(np.zeros((0,)))
  assert t.spec().to_json(include_defaults=False) == {
      "driver": "array",
      "dtype": "float64",
      "array": [],
      "transform": {
          "input_inclusive_min": [0],
          "input_exclusive_max": [0],
          "output": [{}],
      },
  }


@pytest.mark.parametrize("value", [True, False, "open", 100, 1000.5])
@pytest.mark.parametrize(
    "key",
    [
        "recheck_cached_data",
        "recheck_cached_metadata",
        "recheck_cached",
    ],
)
async def test_recheck_cached(key, value):
  spec = ts.Spec({
      "driver": "zarr",
      "kvstore": "memory://",
      "schema": {"dtype": "uint32", "domain": {"shape": [100, 200]}},
  })

  def validate_spec(s):
    j = s.to_json(include_defaults=True)
    if key != "recheck_cached_data":
      assert j["recheck_cached_metadata"] == value
    if key != "recheck_cached_metadata":
      assert j["recheck_cached_data"] == value

  spec_copy = spec.copy()
  spec_copy.update(**{key: value})
  validate_spec(spec_copy)

  t = await ts.open(spec, create=True, **{key: value})
  validate_spec(t.spec())

  t = await ts.open(spec, create=True)
  validate_spec(t.spec(**{key: value}))


async def test_non_utf8_error():
  with pytest.raises(ValueError, match='.*local file "\\\\xfa.*'):
    await ts.open({"driver": "zarr", "kvstore": "file://%fa"})


async def test_write_batch():
  store = await ts.open(
      {"driver": "zarr3", "kvstore": "memory://"},
      dtype=ts.uint8,
      shape=[],
      create=True,
  )
  with pytest.raises(ValueError, match=".*batch can only be specified.*"):
    with ts.Batch() as batch:
      await store.write(42, batch=batch)

  with ts.Batch() as batch:
    write_future = store.write(ts.array(42, dtype=np.uint8), batch=batch)
  await write_future
