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
"""Tests for int2 data type with NumPy."""

# This is based on `bfloat16_test`, which in turn is from Tensorflow:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/lib/core/bfloat16_test.py
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import numpy as np
import pytest
import tensorstore as ts

int2 = ts.int2.type


# Values that should round trip exactly to float and back.
INT2_VALUES = list(range(-2, 1))


@pytest.mark.parametrize("v", INT2_VALUES)
def test_round_trip_to_float(v):
  np.testing.assert_equal(v, float(int2(v)))


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]
)
def test_round_trip_numpy_types(dtype):
  np.testing.assert_equal(dtype(-2), dtype(int2(dtype(-2))))
  np.testing.assert_equal(0, int(int2(dtype(0))))
  np.testing.assert_equal(dtype(1), dtype(int2(np.array(1, dtype))))
  np.testing.assert_equal(
      np.array([-2, 1, -1], int2), int2(np.array([-2, 1, -1], dtype))
  )


@pytest.mark.parametrize("v", INT2_VALUES)
def test_round_trip_to_int(v):
  assert v == int(int2(v))


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]
)
@pytest.mark.parametrize("v", INT2_VALUES)
def test_round_trip_to_numpy(dtype, v):
  np.testing.assert_equal(v, int2(dtype(v)))
  np.testing.assert_equal(v, dtype(int2(dtype(v))))
  np.testing.assert_equal(v, dtype(int2(np.array(v, dtype))))


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]
)
def test_round_trip_to_numpy_array(dtype):
  np.testing.assert_equal(
      np.array(INT2_VALUES, dtype),
      int2(np.array(INT2_VALUES, dtype)).astype(dtype),
  )


@pytest.mark.parametrize("v", INT2_VALUES)
def test_str(v):
  np.testing.assert_equal(str(v), str(int2(v)))


@pytest.mark.parametrize("v", INT2_VALUES)
def test_repr(v):
  np.testing.assert_equal(str(v), repr(int2(v)))


@pytest.mark.parametrize("v", INT2_VALUES)
def test_hash(v):
  """make sure the hash matches that of int."""
  assert hash(int2(v)) == hash(v)


# Tests for Python operations
def test_negate():
  np.testing.assert_equal(1, int(-int2(-1)))
  np.testing.assert_equal(0, int(-int2(0)))
  np.testing.assert_equal(-1, int(-int2(1)))
  np.testing.assert_equal(-2, int(-int2(2)))


def test_add():
  np.testing.assert_equal(0, int(int2(0) + int2(0)))
  np.testing.assert_equal(1, int(int2(1) + int2(0)))
  np.testing.assert_equal(0, int(int2(1) + int2(-1)))
  np.testing.assert_equal(-2, int(int2(1) + int2(1)))


def test_sub():
  np.testing.assert_equal(0, int(int2(0) - int2(0)))
  np.testing.assert_equal(1, int(int2(1) - int2(0)))
  np.testing.assert_equal(-2, int(int2(-1) - int2(1)))


def test_mul():
  np.testing.assert_equal(0, int(int2(0) * int2(0)))
  np.testing.assert_equal(0, float(int2(1) * int2(0)))
  np.testing.assert_equal(-2, float(int2(1) * int2(-2)))


def test_div_mod():
  np.testing.assert_equal(-2, int(int2(-2) // int2(1)))
  np.testing.assert_equal(0, int(int2(-2) % int2(1)))

  np.testing.assert_equal(-1, int(int2(1) // int2(-2)))
  np.testing.assert_equal(-1, int(int2(1) % int2(-2)))


def test_less():
  for v in INT2_VALUES:
    for w in INT2_VALUES:
      assert (v < w) == (int2(v) < int2(w))


def test_less_equal():
  for v in INT2_VALUES:
    for w in INT2_VALUES:
      assert (v <= w) == (int2(v) <= int2(w))


def test_greater():
  for v in INT2_VALUES:
    for w in INT2_VALUES:
      assert (v > w) == (int2(v) > int2(w))


def test_greater_equal():
  for v in INT2_VALUES:
    for w in INT2_VALUES:
      assert (v >= w) == (int2(v) >= int2(w))


def test_equal():
  for v in INT2_VALUES:
    for w in INT2_VALUES:
      assert (v == w) == (int2(v) == int2(w))


def test_not_equal():
  for v in INT2_VALUES:
    for w in INT2_VALUES:
      assert (v != w) == (int2(v) != int2(w))


def test_sort():
  values_to_sort = np.int32(INT2_VALUES)
  sorted_int32 = np.sort(values_to_sort)
  sorted_int2 = np.sort(values_to_sort.astype(int2))
  np.testing.assert_equal(sorted_int32, np.float32(sorted_int2))


def test_dtype():
  assert int2 == np.dtype(int2)


def test_deep_copy_does_not_alter_hash():
  # For context, see https://github.com/google/jax/issues/4651. If the hash
  # value of the type descriptor is not initialized correctly, a deep copy
  # can change the type hash.
  dtype = np.dtype(int2)
  h = hash(dtype)
  _ = copy.deepcopy(dtype)
  assert h == hash(dtype)


def test_array():
  x = np.array([[-2, -1, 0, 1]], dtype=int2)
  assert int2 == x.dtype
  assert "[[-2 -1 0 1]]" == str(x)
  np.testing.assert_equal(x, x)
  assert (x == x).all()


def test_comparisons():
  x = np.array([1, 0, -2], dtype=np.int32)
  bx = x.astype(int2)
  y = np.array([-2, 1, 1], dtype=np.int32)
  by = y.astype(int2)
  np.testing.assert_equal(x == y, bx == by)
  np.testing.assert_equal(x != y, bx != by)
  np.testing.assert_equal(x < y, bx < by)
  np.testing.assert_equal(x > y, bx > by)
  np.testing.assert_equal(x <= y, bx <= by)
  np.testing.assert_equal(x >= y, bx >= by)


def test_equal2():
  a = np.array([-2], int2)
  b = np.array([1], int2)
  assert not a.__eq__(b)


@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.complex64,
        np.complex128,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.intc,
        np.int_,
        np.longlong,
        np.uintc,
        np.ulonglong,
    ],
)
def test_casts(dtype):
  x = np.array([[0, 1]], dtype=dtype)
  y = x.astype(int2)
  z = y.astype(dtype)
  assert np.all(x == y)
  assert int2 == y.dtype
  assert np.all(x == z)
  assert dtype == z.dtype


def test_arange():
  np.testing.assert_equal(
      np.arange(1, dtype=np.int32).astype(int2),
      np.arange(1, dtype=np.int32),
  )
  np.testing.assert_equal(
      np.arange(-2, 1, 2, dtype=np.int32).astype(int2),
      np.arange(-2, 1, 2, dtype=np.int32),
  )
  np.testing.assert_equal(
      np.arange(1, -2, -1, dtype=np.int32).astype(int2),
      np.arange(1, -2, -1, dtype=np.int32),
  )


@pytest.mark.parametrize(
    "op",
    [
        np.negative,
        np.positive,
        np.absolute,
        np.sign,
    ],
    ids=lambda op: op.__name__,
)
@pytest.mark.filterwarnings("ignore:invalid value|divide by zero")
def test_unary_ufunc(op):
  x = np.arange(-1, 2).astype(int2)
  np.testing.assert_equal(op(x).astype(np.int32), op(x.astype(np.int32)))


@pytest.mark.parametrize(
    "op",
    [
        np.add,
        np.subtract,
        np.multiply,
        np.maximum,
        np.minimum,
        np.copysign,
    ],
    ids=lambda op: op.__name__,
)
@pytest.mark.filterwarnings("ignore:invalid value")
def test_binary_ufunc(op):
  rng = np.random.RandomState(seed=42)
  x = rng.randint(-2, 2, 100, dtype=np.int32)
  y = rng.randint(-2, 2, 100, dtype=np.int32)
  ans = op(x, y)
  result = op(x.astype(int2), y.astype(int2)).astype(np.int32)
  mask = (-2 <= ans) & (ans < 2)
  np.testing.assert_equal(ans[mask], result[mask])


@pytest.mark.parametrize(
    "op",
    [
        np.floor_divide,
        np.remainder,
    ],
    ids=lambda op: op.__name__,
)
@pytest.mark.filterwarnings("ignore:invalid value|divide by zero")
def test_binary_ufunc_division(op):
  rng = np.random.RandomState(seed=42)
  x = rng.randint(-2, 2, 100, dtype=np.int32)
  y = rng.randint(-2, 2, 100, dtype=np.int32)
  mask = y != 0
  x = x[mask]
  y = y[mask]
  ans = op(x, y)
  ans = np.where(ans == 2, ans - 4, ans)  # Handle cases where ans == 2.
  result = op(x.astype(int2), y.astype(int2)).astype(np.int32)
  np.testing.assert_equal(ans, result)


@pytest.mark.parametrize(
    "op",
    [
        np.equal,
        np.not_equal,
        np.less,
        np.greater,
        np.less_equal,
        np.greater_equal,
        np.logical_and,
        np.logical_or,
        np.logical_xor,
    ],
    ids=lambda op: op.__name__,
)
def test_binary_predicate_ufunc(op):
  rng = np.random.RandomState(seed=42)
  x = rng.randint(-2, 2, 100).astype(int2)
  y = rng.randint(-2, 2, 100).astype(int2)
  np.testing.assert_equal(op(x, y), op(x.astype(int2), y.astype(int2)))


def test_np_divmod():
  rng = np.random.RandomState(seed=42)
  x = rng.randint(-2, 2, 100, dtype=np.int32)
  y = rng.randint(-2, 2, 100, dtype=np.int32)
  mask = y != 0
  x = x[mask]
  y = y[mask]
  div1, mod1 = np.divmod(x, y)
  div2, mod2 = np.divmod(x.astype(int2), y.astype(int2))
  np.testing.assert_equal(div1, div2.astype(np.int32))
  np.testing.assert_equal(mod1, mod2.astype(np.int32))
