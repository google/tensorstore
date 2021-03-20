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
"""Tests for the bfloat16 NumPy data type."""

# This is based on code from Tensorflow:
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
import itertools
import math

import numpy as np
import pytest
import tensorstore as ts

bfloat16 = ts.bfloat16.type


def numpy_assert_allclose(a, b, **kwargs):
  a = a.astype(np.float32) if a.dtype == bfloat16 else a
  b = b.astype(np.float32) if b.dtype == bfloat16 else b
  return np.testing.assert_allclose(a, b, **kwargs)


epsilon = float.fromhex("1.0p-7")

# Values that should round trip exactly to float and back.
FLOAT_VALUES = [
    0.0, 1.0, -1, 0.5, -0.5, epsilon, 1.0 + epsilon, 1.0 - epsilon,
    -1.0 - epsilon, -1.0 + epsilon, 3.5, 42.0, 255.0, 256.0,
    float("inf"),
    float("-inf"),
    float("nan")
]


@pytest.mark.parametrize("v", FLOAT_VALUES)
def test_round_trip_to_float(v):
  np.testing.assert_equal(v, float(bfloat16(v)))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_round_trip_numpy_types(dtype):
  np.testing.assert_equal(-3.75, dtype(bfloat16(dtype(-3.75))))
  np.testing.assert_equal(1.5, float(bfloat16(dtype(1.5))))
  np.testing.assert_equal(4.5, dtype(bfloat16(np.array(4.5, dtype))))
  np.testing.assert_equal(np.array([2, 5, -1], bfloat16),
                          bfloat16(np.array([2, 5, -1], dtype)))


@pytest.mark.parametrize(
    "v", [-256, -255, -34, -2, -1, 0, 1, 2, 10, 47, 128, 255, 256, 512])
def test_round_trip_to_int(v):
  assert v == int(bfloat16(v))


@pytest.mark.parametrize("dtype",
                         [bfloat16, np.float16, np.float32, np.float64])
@pytest.mark.parametrize("v", FLOAT_VALUES)
def test_round_trip_to_numpy(dtype, v):
  np.testing.assert_equal(v, bfloat16(dtype(v)))
  np.testing.assert_equal(v, dtype(bfloat16(dtype(v))))
  np.testing.assert_equal(v, dtype(bfloat16(np.array(v, dtype))))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_round_trip_to_numpy_array(dtype):
  np.testing.assert_equal(np.array(FLOAT_VALUES, dtype),
                          bfloat16(np.array(FLOAT_VALUES, dtype)).astype(dtype))


def test_str():
  assert "0" == str(bfloat16(0.0))
  assert "1" == str(bfloat16(1.0))
  assert "-3.5" == str(bfloat16(-3.5))
  assert "0.0078125" == str(bfloat16(float.fromhex("1.0p-7")))
  assert "inf" == str(bfloat16(float("inf")))
  assert "-inf" == str(bfloat16(float("-inf")))
  assert "nan" == str(bfloat16(float("nan")))


def test_repr():
  assert "0" == repr(bfloat16(0))
  assert "1" == repr(bfloat16(1))
  assert "-3.5" == repr(bfloat16(-3.5))
  assert "0.0078125" == repr(bfloat16(float.fromhex("1.0p-7")))
  assert "inf" == repr(bfloat16(float("inf")))
  assert "-inf" == repr(bfloat16(float("-inf")))
  assert "nan" == repr(bfloat16(float("nan")))


def test_hash():
  assert 0 == hash(bfloat16(0.0))
  assert 0x3f80 == hash(bfloat16(1.0))


# Tests for Python operations
@pytest.mark.parametrize("v", FLOAT_VALUES)
def test_negate(v):
  np.testing.assert_equal(-v, float(-bfloat16(v)))


def test_add():
  np.testing.assert_equal(0, float(bfloat16(0) + bfloat16(0)))
  np.testing.assert_equal(1, float(bfloat16(1) + bfloat16(0)))
  np.testing.assert_equal(0, float(bfloat16(1) + bfloat16(-1)))
  np.testing.assert_equal(5.5, float(bfloat16(2) + bfloat16(3.5)))
  np.testing.assert_equal(1.25, float(bfloat16(3.5) + bfloat16(-2.25)))
  np.testing.assert_equal(float("inf"),
                          float(bfloat16(float("inf")) + bfloat16(-2.25)))
  np.testing.assert_equal(float("-inf"),
                          float(bfloat16(float("-inf")) + bfloat16(-2.25)))
  assert math.isnan(float(bfloat16(3.5) + bfloat16(float("nan"))))

  # Test type promotion against Numpy scalar values.
  assert np.float32 == type(bfloat16(3.5) + np.float16(2.25))
  assert np.float32 == type(np.float16(3.5) + bfloat16(2.25))
  assert np.float32 == type(bfloat16(3.5) + np.float32(2.25))
  assert np.float32 == type(np.float32(3.5) + bfloat16(2.25))
  assert np.float64 == type(bfloat16(3.5) + np.float64(2.25))
  assert np.float64 == type(np.float64(3.5) + bfloat16(2.25))
  assert np.float64 == type(bfloat16(3.5) + float(2.25))
  assert np.float64 == type(float(3.5) + bfloat16(2.25))
  assert np.float32 == type(bfloat16(3.5) + np.array(2.25, np.float32))
  assert np.float32 == type(np.array(3.5, np.float32) + bfloat16(2.25))


def test_sub():
  np.testing.assert_equal(0, float(bfloat16(0) - bfloat16(0)))
  np.testing.assert_equal(1, float(bfloat16(1) - bfloat16(0)))
  np.testing.assert_equal(2, float(bfloat16(1) - bfloat16(-1)))
  np.testing.assert_equal(-1.5, float(bfloat16(2) - bfloat16(3.5)))
  np.testing.assert_equal(5.75, float(bfloat16(3.5) - bfloat16(-2.25)))
  np.testing.assert_equal(float("-inf"),
                          float(bfloat16(-2.25) - bfloat16(float("inf"))))
  np.testing.assert_equal(float("inf"),
                          float(bfloat16(-2.25) - bfloat16(float("-inf"))))
  assert math.isnan(float(bfloat16(3.5) - bfloat16(float("nan"))))


def test_mul():
  np.testing.assert_equal(0, float(bfloat16(0) * bfloat16(0)))
  np.testing.assert_equal(0, float(bfloat16(1) * bfloat16(0)))
  np.testing.assert_equal(-1, float(bfloat16(1) * bfloat16(-1)))
  np.testing.assert_equal(-7.875, float(bfloat16(3.5) * bfloat16(-2.25)))
  np.testing.assert_equal(float("-inf"),
                          float(bfloat16(float("inf")) * bfloat16(-2.25)))
  np.testing.assert_equal(float("inf"),
                          float(bfloat16(float("-inf")) * bfloat16(-2.25)))
  assert math.isnan(float(bfloat16(3.5) * bfloat16(float("nan"))))


def test_div():
  assert math.isnan(float(bfloat16(0) / bfloat16(0)))
  np.testing.assert_equal(float("inf"), float(bfloat16(1) / bfloat16(0)))
  np.testing.assert_equal(-1, float(bfloat16(1) / bfloat16(-1)))
  np.testing.assert_equal(-1.75, float(bfloat16(3.5) / bfloat16(-2)))
  np.testing.assert_equal(float("-inf"),
                          float(bfloat16(float("inf")) / bfloat16(-2.25)))
  np.testing.assert_equal(float("inf"),
                          float(bfloat16(float("-inf")) / bfloat16(-2.25)))
  assert math.isnan(float(bfloat16(3.5) / bfloat16(float("nan"))))


def test_less():
  for v in FLOAT_VALUES:
    for w in FLOAT_VALUES:
      assert (v < w) == (bfloat16(v) < bfloat16(w))


def test_less_equal():
  for v in FLOAT_VALUES:
    for w in FLOAT_VALUES:
      assert (v <= w) == (bfloat16(v) <= bfloat16(w))


def test_greater():
  for v in FLOAT_VALUES:
    for w in FLOAT_VALUES:
      assert (v > w) == (bfloat16(v) > bfloat16(w))


def test_greater_equal():
  for v in FLOAT_VALUES:
    for w in FLOAT_VALUES:
      assert (v >= w) == (bfloat16(v) >= bfloat16(w))


def test_equal():
  for v in FLOAT_VALUES:
    for w in FLOAT_VALUES:
      assert (v == w) == (bfloat16(v) == bfloat16(w))


def test_not_equal():
  for v in FLOAT_VALUES:
    for w in FLOAT_VALUES:
      assert (v != w) == (bfloat16(v) != bfloat16(w))


def test_nan():
  a = np.isnan(bfloat16(float("nan")))
  assert a
  numpy_assert_allclose(np.array([1.0, a]), np.array([1.0, a]))

  a = np.array([bfloat16(1.34375),
                bfloat16(1.4375),
                bfloat16(float("nan"))], dtype=bfloat16)
  b = np.array([bfloat16(1.3359375),
                bfloat16(1.4375),
                bfloat16(float("nan"))], dtype=bfloat16)
  numpy_assert_allclose(a, b, rtol=0.1, atol=0.1, equal_nan=True, err_msg="",
                        verbose=True)


def test_sort():
  values_to_sort = np.float32(FLOAT_VALUES)
  sorted_f32 = np.sort(values_to_sort)
  sorted_bf16 = np.sort(values_to_sort.astype(bfloat16))
  np.testing.assert_equal(sorted_f32, np.float32(sorted_bf16))


def test_dtype():
  assert bfloat16 == np.dtype(bfloat16)


def test_deep_copy_does_not_alter_hash():
  # For context, see https://github.com/google/jax/issues/4651. If the hash
  # value of the type descriptor is not initialized correctly, a deep copy
  # can change the type hash.
  dtype = np.dtype(bfloat16)
  h = hash(dtype)
  _ = copy.deepcopy(dtype)
  assert h == hash(dtype)


def test_array():
  x = np.array([[1, 2, 3]], dtype=bfloat16)
  assert bfloat16 == x.dtype
  assert "[[1 2 3]]" == str(x)
  np.testing.assert_equal(x, x)
  numpy_assert_allclose(x, x)
  assert (x == x).all()


def test_comparisons():
  x = np.array([401408, 7, -32], dtype=np.float32)
  bx = x.astype(bfloat16)
  y = np.array([82432, 7, 0], dtype=np.float32)
  by = y.astype(bfloat16)
  np.testing.assert_equal(x == y, bx == by)
  np.testing.assert_equal(x != y, bx != by)
  np.testing.assert_equal(x < y, bx < by)
  np.testing.assert_equal(x > y, bx > by)
  np.testing.assert_equal(x <= y, bx <= by)
  np.testing.assert_equal(x >= y, bx >= by)


def test_equal2():
  a = np.array([401408], bfloat16)
  b = np.array([82432], bfloat16)
  assert not a.__eq__(b)


def test_can_cast():
  allowed_casts = [
      (np.bool_, bfloat16),
      (np.int8, bfloat16),
      (np.uint8, bfloat16),
      (bfloat16, np.float32),
      (bfloat16, np.float64),
      (bfloat16, np.complex64),
      (bfloat16, np.complex128),
  ]
  all_dtypes = [
      np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
      np.complex64, np.complex128, np.uint8, np.uint16, np.uint32, np.uint64,
      np.intc, np.int_, np.longlong, np.uintc, np.ulonglong
  ]
  for d in all_dtypes:
    assert ((bfloat16, d) in allowed_casts) == np.can_cast(bfloat16, d)
    assert ((d, bfloat16) in allowed_casts) == np.can_cast(d, bfloat16)


@pytest.mark.parametrize("dtype", [
    np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
    np.complex64, np.complex128, np.uint8, np.uint16, np.uint32, np.uint64,
    np.intc, np.int_, np.longlong, np.uintc, np.ulonglong
])
def test_casts(dtype):
  x = np.array([[1, 2, 3]], dtype=dtype)
  y = x.astype(bfloat16)
  z = y.astype(dtype)
  assert np.all(x == y)
  assert bfloat16 == y.dtype
  assert np.all(x == z)
  assert dtype == z.dtype


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.filterwarnings("ignore:casting complex values to real")
def test_conform_numpy_complex(dtype):
  x = np.array([1.1, 2.2 + 2.2j, 3.3], dtype=dtype)
  y_np = x.astype(np.float32)
  y_tf = x.astype(bfloat16)
  numpy_assert_allclose(y_np, y_tf, atol=2e-2)

  z_np = y_np.astype(dtype)
  z_tf = y_tf.astype(dtype)
  numpy_assert_allclose(z_np, z_tf, atol=2e-2)


def test_arange():
  np.testing.assert_equal(
      np.arange(100, dtype=np.float32).astype(bfloat16),
      np.arange(100, dtype=bfloat16))
  np.testing.assert_equal(
      np.arange(-10.5, 7.8, 0.5, dtype=np.float32).astype(bfloat16),
      np.arange(-10.5, 7.8, 0.5, dtype=bfloat16))
  np.testing.assert_equal(
      np.arange(-0., -7., -0.25, dtype=np.float32).astype(bfloat16),
      np.arange(-0., -7., -0.25, dtype=bfloat16))
  np.testing.assert_equal(
      np.arange(-16384., 16384., 64., dtype=np.float32).astype(bfloat16),
      np.arange(-16384., 16384., 64., dtype=bfloat16))


@pytest.mark.parametrize("op", [
    np.negative, np.positive, np.absolute, np.fabs, np.rint, np.sign,
    np.conjugate, np.exp, np.exp2, np.expm1, np.log, np.log10, np.log1p,
    np.log2, np.sqrt, np.square, np.cbrt, np.reciprocal, np.sin, np.cos, np.tan,
    np.arcsin, np.arccos, np.arctan, np.sinh, np.cosh, np.tanh, np.arcsinh,
    np.arccosh, np.arctanh, np.deg2rad, np.rad2deg, np.floor, np.ceil, np.trunc
], ids=lambda op: op.__name__)
@pytest.mark.filterwarnings("ignore:invalid value|divide by zero")
def test_unary_ufunc(op):
  rng = np.random.RandomState(seed=42)
  x = rng.randn(3, 7, 10).astype(bfloat16)
  numpy_assert_allclose(
      op(x).astype(np.float32), op(x.astype(np.float32)), rtol=1e-2)


@pytest.mark.parametrize("op", [
    np.add, np.subtract, np.multiply, np.divide, np.logaddexp, np.logaddexp2,
    np.floor_divide, np.power, np.remainder, np.fmod, np.heaviside, np.arctan2,
    np.hypot, np.maximum, np.minimum, np.fmax, np.fmin, np.copysign
], ids=lambda op: op.__name__)
@pytest.mark.filterwarnings("ignore:invalid value")
def test_binary_ufunc(op):
  rng = np.random.RandomState(seed=42)
  x = rng.randn(3, 7, 10).astype(bfloat16)
  y = rng.randn(4, 1, 7, 10).astype(bfloat16)
  numpy_assert_allclose(
      op(x, y).astype(np.float32), op(x.astype(np.float32),
                                      y.astype(np.float32)), rtol=1e-2)


@pytest.mark.parametrize("op", [
    np.equal, np.not_equal, np.less, np.greater, np.less_equal,
    np.greater_equal, np.logical_and, np.logical_or, np.logical_xor
], ids=lambda op: op.__name__)
def test_binary_predicate_ufunc(op):
  rng = np.random.RandomState(seed=42)
  x = rng.randn(3, 7).astype(bfloat16)
  y = rng.randn(4, 1, 7).astype(bfloat16)
  np.testing.assert_equal(op(x, y),
                          op(x.astype(np.float32), y.astype(np.float32)))


@pytest.mark.parametrize(
    "op", [np.isfinite, np.isinf, np.isnan, np.signbit, np.logical_not],
    ids=lambda op: op.__name__)
def test_predicate_ufunc(op):
  rng = np.random.RandomState(seed=42)
  shape = (3, 7, 10)
  posinf_flips = rng.rand(*shape) < 0.1
  neginf_flips = rng.rand(*shape) < 0.1
  nan_flips = rng.rand(*shape) < 0.1
  vals = rng.randn(*shape)
  vals = np.where(posinf_flips, np.inf, vals)
  vals = np.where(neginf_flips, -np.inf, vals)
  vals = np.where(nan_flips, np.nan, vals)
  vals = vals.astype(bfloat16)
  np.testing.assert_equal(op(vals), op(vals.astype(np.float32)))


def test_divmod():
  rng = np.random.RandomState(seed=42)
  x = rng.randn(3, 7).astype(bfloat16)
  y = rng.randn(4, 1, 7).astype(bfloat16)
  o1, o2 = np.divmod(x, y)
  e1, e2 = np.divmod(x.astype(np.float32), y.astype(np.float32))
  numpy_assert_allclose(o1, e1, rtol=1e-2)
  numpy_assert_allclose(o2, e2, rtol=1e-2)


def test_modf():
  rng = np.random.RandomState(seed=42)
  x = rng.randn(3, 7).astype(bfloat16)
  o1, o2 = np.modf(x)
  e1, e2 = np.modf(x.astype(np.float32))
  numpy_assert_allclose(o1.astype(np.float32), e1, rtol=1e-2)
  numpy_assert_allclose(o2.astype(np.float32), e2, rtol=1e-2)


def test_ldexp():
  rng = np.random.RandomState(seed=42)
  x = rng.randn(3, 7).astype(bfloat16)
  y = rng.randint(-50, 50, (1, 7))
  numpy_assert_allclose(
      np.ldexp(x, y).astype(np.float32), np.ldexp(x.astype(np.float32), y),
      rtol=1e-2, atol=1e-6)


def test_frexp():
  rng = np.random.RandomState(seed=42)
  x = rng.randn(3, 7).astype(bfloat16)
  mant1, exp1 = np.frexp(x)
  mant2, exp2 = np.frexp(x.astype(np.float32))
  np.testing.assert_equal(exp1, exp2)
  numpy_assert_allclose(mant1, mant2, rtol=1e-2)


def test_nextafter():
  one = np.array(1., dtype=bfloat16)
  two = np.array(2., dtype=bfloat16)
  zero = np.array(0., dtype=bfloat16)
  nan = np.array(np.nan, dtype=bfloat16)
  np.testing.assert_equal(np.nextafter(one, two) - one, epsilon)
  np.testing.assert_equal(np.nextafter(one, zero) - one, -epsilon / 2)
  np.testing.assert_equal(np.isnan(np.nextafter(nan, one)), True)
  np.testing.assert_equal(np.isnan(np.nextafter(one, nan)), True)
  np.testing.assert_equal(np.nextafter(one, one), one)
  smallest_denormal = float.fromhex("1.0p-133")
  np.testing.assert_equal(np.nextafter(zero, one), smallest_denormal)
  np.testing.assert_equal(np.nextafter(zero, -one), -smallest_denormal)
  for a, b in itertools.permutations([0., -0., nan], 2):
    np.testing.assert_equal(
        np.nextafter(np.array(a, dtype=np.float32),
                     np.array(b, dtype=np.float32)),
        np.nextafter(np.array(a, dtype=bfloat16), np.array(b, dtype=bfloat16)))
