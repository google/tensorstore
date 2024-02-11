# Copyright 2023 The TensorStore Authors
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
"""Tests for the int4 NumPy data type."""

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

int4 = ts.int4.type


# Values that should round trip exactly to float and back.
INT4_VALUES = list(range(-8, 8))


@pytest.mark.parametrize("v", INT4_VALUES)
def test_round_trip_to_float(v):
  np.testing.assert_equal(v, float(int4(v)))


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]
)
def test_round_trip_numpy_types(dtype):
  np.testing.assert_equal(dtype(-8), dtype(int4(dtype(-8))))
  np.testing.assert_equal(0, int(int4(dtype(0))))
  np.testing.assert_equal(dtype(7), dtype(int4(np.array(7, dtype))))
  np.testing.assert_equal(
      np.array([2, 5, -1], int4), int4(np.array([2, 5, -1], dtype))
  )


@pytest.mark.parametrize("v", INT4_VALUES)
def test_round_trip_to_int(v):
  assert v == int(int4(v))


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]
)
@pytest.mark.parametrize("v", INT4_VALUES)
def test_round_trip_to_numpy(dtype, v):
  np.testing.assert_equal(v, int4(dtype(v)))
  np.testing.assert_equal(v, dtype(int4(dtype(v))))
  np.testing.assert_equal(v, dtype(int4(np.array(v, dtype))))


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]
)
def test_round_trip_to_numpy_array(dtype):
  np.testing.assert_equal(
      np.array(INT4_VALUES, dtype),
      int4(np.array(INT4_VALUES, dtype)).astype(dtype),
  )


@pytest.mark.parametrize("v", INT4_VALUES)
def test_str(v):
  np.testing.assert_equal(str(v), str(int4(v)))


@pytest.mark.parametrize("v", INT4_VALUES)
def test_repr(v):
  np.testing.assert_equal(str(v), repr(int4(v)))


@pytest.mark.parametrize("v", INT4_VALUES)
def test_hash(v):
  """make sure the hash matches that of int."""
  assert hash(int4(v)) == hash(v)


# Tests for Python operations
def test_negate():
  np.testing.assert_equal(-8, int(-int4(-8)))
  np.testing.assert_equal(7, int(-int4(-7)))
  np.testing.assert_equal(6, int(-int4(-6)))
  np.testing.assert_equal(5, int(-int4(-5)))
  np.testing.assert_equal(4, int(-int4(-4)))
  np.testing.assert_equal(3, int(-int4(-3)))
  np.testing.assert_equal(2, int(-int4(-2)))
  np.testing.assert_equal(1, int(-int4(-1)))
  np.testing.assert_equal(0, int(-int4(0)))
  np.testing.assert_equal(-1, int(-int4(1)))
  np.testing.assert_equal(-2, int(-int4(2)))
  np.testing.assert_equal(-3, int(-int4(3)))
  np.testing.assert_equal(-4, int(-int4(4)))
  np.testing.assert_equal(-5, int(-int4(5)))
  np.testing.assert_equal(-6, int(-int4(6)))
  np.testing.assert_equal(-7, int(-int4(7)))


def test_add():
  np.testing.assert_equal(0, int(int4(0) + int4(0)))
  np.testing.assert_equal(1, int(int4(1) + int4(0)))
  np.testing.assert_equal(0, int(int4(1) + int4(-1)))
  np.testing.assert_equal(-2, int(int4(7) + int4(7)))


def test_sub():
  np.testing.assert_equal(0, int(int4(0) - int4(0)))
  np.testing.assert_equal(1, int(int4(1) - int4(0)))
  np.testing.assert_equal(2, int(int4(1) - int4(-1)))
  np.testing.assert_equal(-1, int(int4(7) - int4(-8)))


def test_mul():
  np.testing.assert_equal(0, int(int4(0) * int4(0)))
  np.testing.assert_equal(0, float(int4(1) * int4(0)))
  np.testing.assert_equal(-8, float(int4(4) * int4(-2)))
  np.testing.assert_equal(6, float(int4(2) * int4(3)))


def test_div_mod():
  np.testing.assert_equal(2, int(int4(6) // int4(3)))
  np.testing.assert_equal(0, int(int4(6) % int4(3)))

  np.testing.assert_equal(-2, int(int4(6) // int4(-3)))
  np.testing.assert_equal(0, int(int4(6) % int4(-3)))

  np.testing.assert_equal(-2, int(int4(-6) // int4(3)))
  np.testing.assert_equal(0, int(int4(-6) % int4(3)))

  np.testing.assert_equal(2, int(int4(-6) // int4(-3)))
  np.testing.assert_equal(0, int(int4(-6) % int4(-3)))

  np.testing.assert_equal(2, int(int4(7) // int4(3)))
  np.testing.assert_equal(1, int(int4(7) % int4(3)))

  np.testing.assert_equal(-3, int(int4(7) // int4(-3)))
  np.testing.assert_equal(-2, int(int4(7) % int4(-3)))

  np.testing.assert_equal(-3, int(int4(-7) // int4(3)))
  np.testing.assert_equal(2, int(int4(-7) % int4(3)))

  np.testing.assert_equal(2, int(int4(-7) // int4(-3)))
  np.testing.assert_equal(-1, int(int4(-7) % int4(-3)))


def test_less():
  for v in INT4_VALUES:
    for w in INT4_VALUES:
      assert (v < w) == (int4(v) < int4(w))


def test_less_equal():
  for v in INT4_VALUES:
    for w in INT4_VALUES:
      assert (v <= w) == (int4(v) <= int4(w))


def test_greater():
  for v in INT4_VALUES:
    for w in INT4_VALUES:
      assert (v > w) == (int4(v) > int4(w))


def test_greater_equal():
  for v in INT4_VALUES:
    for w in INT4_VALUES:
      assert (v >= w) == (int4(v) >= int4(w))


def test_equal():
  for v in INT4_VALUES:
    for w in INT4_VALUES:
      assert (v == w) == (int4(v) == int4(w))


def test_not_equal():
  for v in INT4_VALUES:
    for w in INT4_VALUES:
      assert (v != w) == (int4(v) != int4(w))


def test_sort():
  values_to_sort = np.int32(INT4_VALUES)
  sorted_int32 = np.sort(values_to_sort)
  sorted_int4 = np.sort(values_to_sort.astype(int4))
  np.testing.assert_equal(sorted_int32, np.float32(sorted_int4))


def test_dtype():
  assert int4 == np.dtype(int4)


def test_deep_copy_does_not_alter_hash():
  # For context, see https://github.com/google/jax/issues/4651. If the hash
  # value of the type descriptor is not initialized correctly, a deep copy
  # can change the type hash.
  dtype = np.dtype(int4)
  h = hash(dtype)
  _ = copy.deepcopy(dtype)
  assert h == hash(dtype)


def test_array():
  x = np.array([[1, 2, 3]], dtype=int4)
  assert int4 == x.dtype
  assert "[[1 2 3]]" == str(x)
  np.testing.assert_equal(x, x)
  assert (x == x).all()


def test_comparisons():
  x = np.array([7, 0, -8], dtype=np.int32)
  bx = x.astype(int4)
  y = np.array([6, 0, -5], dtype=np.int32)
  by = y.astype(int4)
  np.testing.assert_equal(x == y, bx == by)
  np.testing.assert_equal(x != y, bx != by)
  np.testing.assert_equal(x < y, bx < by)
  np.testing.assert_equal(x > y, bx > by)
  np.testing.assert_equal(x <= y, bx <= by)
  np.testing.assert_equal(x >= y, bx >= by)


def test_equal2():
  a = np.array([-8], int4)
  b = np.array([7], int4)
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
  x = np.array([[1, 2, 3]], dtype=dtype)
  y = x.astype(int4)
  z = y.astype(dtype)
  assert np.all(x == y)
  assert int4 == y.dtype
  assert np.all(x == z)
  assert dtype == z.dtype


def test_arange():
  np.testing.assert_equal(
      np.arange(7, dtype=np.int32).astype(int4),
      np.arange(7, dtype=np.int32),
  )
  np.testing.assert_equal(
      np.arange(-8, 7, 2, dtype=np.int32).astype(int4),
      np.arange(-8, 7, 2, dtype=np.int32),
  )
  np.testing.assert_equal(
      np.arange(4, -8, -3, dtype=np.int32).astype(int4),
      np.arange(4, -8, -3, dtype=np.int32),
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
  x = np.arange(-7, 8).astype(int4)
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
  x = rng.randint(-8, 8, 100, dtype=np.int32)
  y = rng.randint(-8, 8, 100, dtype=np.int32)
  ans = op(x, y)
  result = op(x.astype(int4), y.astype(int4)).astype(np.int32)
  mask = (-8 <= ans) & (ans < 8)
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
  x = rng.randint(-8, 8, 100, dtype=np.int32)
  y = rng.randint(-8, 8, 100, dtype=np.int32)
  mask = y != 0
  x = x[mask]
  y = y[mask]
  ans = op(x, y)
  result = op(x.astype(int4), y.astype(int4)).astype(np.int32)
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
  x = rng.randint(-8, 8, 100).astype(int4)
  y = rng.randint(-8, 8, 100).astype(int4)
  np.testing.assert_equal(op(x, y), op(x.astype(int4), y.astype(int4)))


def test_np_divmod():
  rng = np.random.RandomState(seed=42)
  x = rng.randint(-8, 8, 100, dtype=np.int32)
  y = rng.randint(-8, 8, 100, dtype=np.int32)
  mask = y != 0
  x = x[mask]
  y = y[mask]
  div1, mod1 = np.divmod(x, y)
  div2, mod2 = np.divmod(x.astype(int4), y.astype(int4))
  np.testing.assert_equal(div1, div2.astype(np.int32))
  np.testing.assert_equal(mod1, mod2.astype(np.int32))
