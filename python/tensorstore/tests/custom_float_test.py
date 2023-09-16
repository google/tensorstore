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

"""Test cases for custom floating point types."""

# This is based on code from ml_dtypes:
# https://github.com/jax-ml/ml_dtypes/blob/main/ml_dtypes/tests/custom_float_test.py
#
# Copyright 2022 The ml_dtypes Authors.
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


import collections
import contextlib
import copy
import itertools
import math
import pickle
import sys
from typing import Type
import warnings
import numpy as np
import pytest
import tensorstore

bfloat16 = tensorstore.bfloat16.type
float8_e4m3b11fnuz = tensorstore.float8_e4m3b11fnuz.type
float8_e4m3fn = tensorstore.float8_e4m3fn.type
float8_e4m3fnuz = tensorstore.float8_e4m3fnuz.type
float8_e5m2 = tensorstore.float8_e5m2.type
float8_e5m2fnuz = tensorstore.float8_e5m2fnuz.type


class _Bfloat16Defines:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-126")
    self.smallest_normal = bfloat16(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-133")
    self.smallest_subnormal = bfloat16(smallest_subnormal)
    eps = float.fromhex("0x1p-7")
    self.eps = bfloat16(eps)
    max_ = float.fromhex("0x1.FEp127")
    self.max = bfloat16(max_)


class _Float8E4m3b11fnuzDefines:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-10")
    self.smallest_normal = float8_e4m3b11fnuz(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-13")
    self.smallest_subnormal = float8_e4m3b11fnuz(smallest_subnormal)
    eps = float.fromhex("0x1p-3")
    self.eps = float8_e4m3b11fnuz(eps)
    max_ = float.fromhex("0x1.Ep4")
    self.max = float8_e4m3b11fnuz(max_)


class _Float8E4m3fnDefines:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-6")
    self.smallest_normal = float8_e4m3fn(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-9")
    self.smallest_subnormal = float8_e4m3fn(smallest_subnormal)
    eps = float.fromhex("0x1p-3")
    self.eps = float8_e4m3fn(eps)
    max_ = float.fromhex("0x1.Cp8")
    self.max = float8_e4m3fn(max_)


class _Float8E4m3fnuzDefines:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-7")
    self.smallest_normal = float8_e4m3fnuz(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-10")
    self.smallest_subnormal = float8_e4m3fnuz(smallest_subnormal)
    eps = float.fromhex("0x1p-3")
    self.eps = float8_e4m3fnuz(eps)
    max_ = float.fromhex("0x1.Ep7")
    self.max = float8_e4m3fnuz(max_)


class _Float8E5m2Defines:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-14")
    self.smallest_normal = float8_e5m2(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-16")
    self.smallest_subnormal = float8_e5m2(smallest_subnormal)
    eps = float.fromhex("0x1p-2")
    self.eps = float8_e5m2(eps)
    max_ = float.fromhex("0x1.Cp15")
    self.max = float8_e5m2(max_)


class _Float8E5m2fnuzDefines:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-15")
    self.smallest_normal = float8_e5m2fnuz(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-17")
    self.smallest_subnormal = float8_e5m2fnuz(smallest_subnormal)
    eps = float.fromhex("0x1p-2")
    self.eps = float8_e5m2fnuz(eps)
    max_ = float.fromhex("0x1.Cp15")
    self.max = float8_e5m2fnuz(max_)


_FINFO = {
    bfloat16: _Bfloat16Defines(),
    float8_e4m3fn: _Float8E4m3fnDefines(),
    float8_e4m3fnuz: _Float8E4m3fnuzDefines(),
    float8_e4m3b11fnuz: _Float8E4m3b11fnuzDefines(),
    float8_e5m2: _Float8E5m2Defines(),
    float8_e5m2fnuz: _Float8E5m2fnuzDefines(),
}


def _finfo(dtype):
  return _FINFO[dtype]


@contextlib.contextmanager
def ignore_warning(**kw):
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", **kw)
    yield


def numpy_assert_allclose(a, b, float_type, **kwargs):
  a = a.astype(np.float32) if a.dtype == float_type else a
  b = b.astype(np.float32) if b.dtype == float_type else b
  return np.testing.assert_allclose(a, b, **kwargs)


def numpy_promote_types(
    a: Type[np.generic],
    b: Type[np.generic],
    float_type: Type[np.generic],
    next_largest_fp_type: Type[np.generic],
) -> Type[np.generic]:
  if a == float_type and b == float_type:
    return float_type
  if a == float_type:
    a = next_largest_fp_type
  if b == float_type:
    b = next_largest_fp_type
  return np.promote_types(a, b)


def truncate(x, float_type):
  if isinstance(x, np.ndarray):
    return x.astype(float_type).astype(np.float32)
  else:
    return type(x)(float_type(x))


def binary_operation_test(a, b, op, float_type):
  a = float_type(a)
  b = float_type(b)
  expected = op(np.float32(a), np.float32(b))
  result = op(a, b)
  if math.isnan(expected):
    if not math.isnan(result):
      raise AssertionError("%s expected to be nan." % repr(result))
  else:
    np.testing.assert_equal(
        truncate(expected, float_type=float_type), float(result)
    )


def dtype_has_inf(dtype):
  """Determines if the dtype has an `inf` representation."""
  inf = float("inf")
  is_inf = False
  try:
    x = dtype(inf)
    is_inf = np.isinf(x)
  except (OverflowError, ValueError):
    pass
  return is_inf


FLOAT_DTYPES = [
    bfloat16,
    float8_e4m3b11fnuz,
    float8_e4m3fn,
    float8_e4m3fnuz,
    float8_e5m2,
    float8_e5m2fnuz,
]

# Values that should round trip exactly to float and back.
# pylint: disable=g-complex-comprehension
FLOAT_VALUES = {
    dtype: [
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        float(_finfo(dtype).eps),
        1.0 + float(_finfo(dtype).eps),
        1.0 - float(_finfo(dtype).eps),
        -1.0 - float(_finfo(dtype).eps),
        -1.0 + float(_finfo(dtype).eps),
        3.5,
        4,
        5,
        7,
        float(_finfo(dtype).max),
        -float(_finfo(dtype).max),
        float("nan"),
        float("-nan"),
        float("inf") if dtype_has_inf(dtype) else 0.0,
        float("-inf") if dtype_has_inf(dtype) else 0.0,
    ]
    for dtype in FLOAT_DTYPES
}

# Values that should round trip exactly to integer and back.
INT_VALUES = {
    bfloat16: [0, 1, 2, 10, 34, 47, 128, 255, 256, 512],
    float8_e4m3b11fnuz: [*range(16), *range(16, 30, 2)],
    float8_e4m3fn: list(
        itertools.chain.from_iterable(
            range(1 << n, 2 << n, 1 << max(0, n - 3)) for n in range(9)
        )
    )[:-1],
    float8_e4m3fnuz: list(
        itertools.chain.from_iterable(
            range(1 << n, 2 << n, 1 << max(0, n - 3)) for n in range(8)
        )
    )[:-1],
    float8_e5m2: list(
        itertools.chain.from_iterable(
            range(1 << n, 2 << n, 1 << max(0, n - 2)) for n in range(16)
        )
    ),
    float8_e5m2fnuz: list(
        itertools.chain.from_iterable(
            range(1 << n, 2 << n, 1 << max(0, n - 2)) for n in range(16)
        )
    ),
}

BITS_TYPE = {
    bfloat16: np.uint16,
    float8_e4m3b11fnuz: np.uint8,
    float8_e4m3fn: np.uint8,
    float8_e4m3fnuz: np.uint8,
    float8_e5m2: np.uint8,
    float8_e5m2fnuz: np.uint8,
}

# parametrize all tests
pytestmark = pytest.mark.parametrize(
    "float_type", [dtype for dtype in FLOAT_DTYPES]
)

"""Tests the non-numpy Python methods of the custom float type."""


def test_module_name(float_type):
  assert float_type.__module__ == "tensorstore"


@pytest.mark.skip(
    "TODO(ChromeHearts) error: _pickle.PicklingError: "
    "Can't pickle <class 'tensorstore.bfloat16'>: "
    "it's not the same object as tensorstore.bfloat16"
)
def test_pickleable(float_type):
  # https://github.com/google/jax/discussions/8505
  x = np.arange(10, dtype=float_type)
  serialized = pickle.dumps(x)
  x_out = pickle.loads(serialized)
  assert x_out.dtype == x.dtype
  np.testing.assert_array_equal(x_out.astype("float32"), x.astype("float32"))


def test_round_trip_to_float(float_type):
  for v in FLOAT_VALUES[float_type]:
    np.testing.assert_equal(v, float(float_type(v)))


@ignore_warning(category=RuntimeWarning, message="overflow encountered")
def test_round_trip_numpy_types(float_type):
  for dtype in [np.float16, np.float32, np.float64, np.longdouble]:
    for f in FLOAT_VALUES[float_type]:
      np.testing.assert_equal(dtype(f), dtype(float_type(dtype(f))))
      np.testing.assert_equal(float(dtype(f)), float(float_type(dtype(f))))
      np.testing.assert_equal(dtype(f), dtype(float_type(np.array(f, dtype))))

    np.testing.assert_equal(
        dtype(np.array(FLOAT_VALUES[float_type], float_type)),
        np.array(FLOAT_VALUES[float_type], dtype),
    )


def test_round_trip_to_int(float_type):
  for v in INT_VALUES[float_type]:
    assert v == int(float_type(v))
    assert -v == int(float_type(-v))


@ignore_warning(category=RuntimeWarning, message="overflow encountered")
def test_round_trip_to_numpy(float_type):
  for dtype in [
      float_type,
      np.float16,
      np.float32,
      np.float64,
      np.longdouble,
  ]:
    for v in FLOAT_VALUES[float_type]:
      np.testing.assert_equal(dtype(v), dtype(float_type(dtype(v))))
      np.testing.assert_equal(dtype(v), dtype(float_type(dtype(v))))
      np.testing.assert_equal(dtype(v), dtype(float_type(np.array(v, dtype))))
    if dtype != float_type:
      np.testing.assert_equal(
          np.array(FLOAT_VALUES[float_type], dtype),
          float_type(np.array(FLOAT_VALUES[float_type], dtype)).astype(dtype),
      )


def test_between_custom_types(float_type):
  for dtype in FLOAT_DTYPES:
    x = np.array(FLOAT_VALUES[float_type], dtype=dtype)
    y = x.astype(float_type)
    z = x.astype(float).astype(float_type)
    numpy_assert_allclose(y, z, float_type=float_type)


def test_str(float_type):
  for value in FLOAT_VALUES[float_type]:
    assert "%.6g" % float(float_type(value)) == str(float_type(value))


def test_from_str(float_type):
  assert float_type(1.2) == float_type("1.2")
  assert np.isnan(float_type("nan"))
  assert np.isnan(float_type("-nan"))
  if dtype_has_inf(float_type):
    assert float_type(float("inf")) == float_type("inf")
    assert float_type(float("-inf")) == float_type("-inf")


def test_repr(float_type):
  for value in FLOAT_VALUES[float_type]:
    assert "%.6g" % float(float_type(value)) == repr(float_type(value))


def test_item(float_type):
  assert isinstance(float_type(0).item(), float)


def test_hash_zero(float_type):
  """Tests that negative zero and zero hash to the same value."""
  assert hash(float_type(-0.0)) == hash(float_type(0.0))


def test_hash_numbers(float_type):
  for value in np.extract(
      np.isfinite(FLOAT_VALUES[float_type]), FLOAT_VALUES[float_type]
  ):
    assert hash(value) == hash(float_type(value))


def test_hash_nan(float_type):
  for nan in [
      float_type(float("nan")),
      float_type(float("-nan")),
  ]:
    nan_hash = hash(nan)
    nan_object_hash = object.__hash__(nan)
    # The hash of a NaN is either 0 or a hash of the object pointer.
    assert nan_hash in (sys.hash_info.nan, nan_object_hash)


def test_hash_inf(float_type):
  if dtype_has_inf(float_type):
    assert sys.hash_info.inf == hash(float_type(float("inf")))
    assert -sys.hash_info.inf == hash(float_type(float("-inf")))


# Tests for Python operations
def test_negate(float_type):
  for v in FLOAT_VALUES[float_type]:
    np.testing.assert_equal(
        float(float_type(-float(float_type(v)))), float(-float_type(v))
    )


def test_add(float_type):
  for a, b in [
      (0, 0),
      (1, 0),
      (1, -1),
      (2, 3.5),
      (3.5, -2.25),
      (float("inf"), -2.25),
      (float("-inf"), -2.25),
      (3.5, float("nan")),
  ]:
    binary_operation_test(a, b, op=lambda a, b: a + b, float_type=float_type)


def test_add_scalar_type_promotion(float_type):
  """Tests type promotion against Numpy scalar values."""
  types = [float_type, np.float16, np.float32, np.float64, np.longdouble]
  for lhs_type in types:
    for rhs_type in types:
      expected_type = numpy_promote_types(
          lhs_type,
          rhs_type,
          float_type=float_type,
          next_largest_fp_type=np.float32,
      )
      actual_type = type(lhs_type(3.5) + rhs_type(2.25))
      assert expected_type == actual_type


def test_add_array_type_promotion(float_type):
  assert np.float32 == type(float_type(3.5) + np.array(2.25, np.float32))
  assert np.float32 == type(np.array(3.5, np.float32) + float_type(2.25))


def test_sub(float_type):
  for a, b in [
      (0, 0),
      (1, 0),
      (1, -1),
      (2, 3.5),
      (3.5, -2.25),
      (-2.25, float("inf")),
      (-2.25, float("-inf")),
      (3.5, float("nan")),
  ]:
    binary_operation_test(a, b, op=lambda a, b: a - b, float_type=float_type)


def test_mul(float_type):
  for a, b in [
      (0, 0),
      (1, 0),
      (1, -1),
      (3.5, -2.25),
      (float("inf"), -2.25),
      (float("-inf"), -2.25),
      (3.5, float("nan")),
  ]:
    binary_operation_test(a, b, op=lambda a, b: a * b, float_type=float_type)


@ignore_warning(category=RuntimeWarning, message="invalid value encountered")
@ignore_warning(category=RuntimeWarning, message="divide by zero encountered")
def test_div(float_type):
  for a, b in [
      (0, 0),
      (1, 0),
      (1, -1),
      (2, 3.5),
      (3.5, -2.25),
      (float("inf"), -2.25),
      (float("-inf"), -2.25),
      (3.5, float("nan")),
  ]:
    binary_operation_test(a, b, op=lambda a, b: a / b, float_type=float_type)


def test_less(float_type):
  for v in FLOAT_VALUES[float_type]:
    for w in FLOAT_VALUES[float_type]:
      assert (v < w) == (float_type(v) < float_type(w))


def test_less_equal(float_type):
  for v in FLOAT_VALUES[float_type]:
    for w in FLOAT_VALUES[float_type]:
      assert (v <= w) == (float_type(v) <= float_type(w))


def test_greater(float_type):
  for v in FLOAT_VALUES[float_type]:
    for w in FLOAT_VALUES[float_type]:
      assert (v > w) == (float_type(v) > float_type(w))


def test_greater_equal(float_type):
  for v in FLOAT_VALUES[float_type]:
    for w in FLOAT_VALUES[float_type]:
      assert (v >= w) == (float_type(v) >= float_type(w))


def test_equal(float_type):
  for v in FLOAT_VALUES[float_type]:
    for w in FLOAT_VALUES[float_type]:
      assert (v == w) == (float_type(v) == float_type(w))


def test_not_equal(float_type):
  for v in FLOAT_VALUES[float_type]:
    for w in FLOAT_VALUES[float_type]:
      assert (v != w) == (float_type(v) != float_type(w))


def test_nan(float_type):
  a = np.isnan(float_type(float("nan")))
  assert a
  numpy_assert_allclose(
      np.array([1.0, a]), np.array([1.0, a]), float_type=float_type
  )

  a = np.array(
      [float_type(1.34375), float_type(1.4375), float_type(float("nan"))],
      dtype=float_type,
  )
  b = np.array(
      [float_type(1.3359375), float_type(1.4375), float_type(float("nan"))],
      dtype=float_type,
  )
  numpy_assert_allclose(
      a,
      b,
      rtol=0.1,
      atol=0.1,
      equal_nan=True,
      err_msg="",
      verbose=True,
      float_type=float_type,
  )


def test_sort(float_type):
  # Note: np.sort doesn't work properly with NaNs since they always compare
  # False.
  values_to_sort = np.float32(
      [x for x in FLOAT_VALUES[float_type] if not np.isnan(x)]
  )
  sorted_f32 = np.sort(values_to_sort)
  sorted_float_type = np.sort(values_to_sort.astype(float_type))  # pylint: disable=too-many-function-args
  np.testing.assert_equal(sorted_f32, np.float32(sorted_float_type))


def test_argmax(float_type):
  values_to_sort = np.float32(float_type(np.float32(FLOAT_VALUES[float_type])))
  argmax_f32 = np.argmax(values_to_sort)
  argmax_float_type = np.argmax(values_to_sort.astype(float_type))  # pylint: disable=too-many-function-args
  np.testing.assert_equal(argmax_f32, argmax_float_type)


def test_argmax_on_nan(float_type):
  """Ensures we return the right thing for multiple NaNs."""
  one_with_nans = np.array([1.0, float("nan"), float("nan")], dtype=np.float32)
  np.testing.assert_equal(
      np.argmax(one_with_nans.astype(float_type)), np.argmax(one_with_nans)
  )


def test_argmax_on_negative_infinity(float_type):
  """Ensures we return the right thing for negative infinities."""
  inf = np.array([float("-inf")], dtype=np.float32)
  np.testing.assert_equal(np.argmax(inf.astype(float_type)), np.argmax(inf))


def test_argmin(float_type):
  values_to_sort = np.float32(float_type(np.float32(FLOAT_VALUES[float_type])))
  argmin_f32 = np.argmin(values_to_sort)
  argmin_float_type = np.argmin(values_to_sort.astype(float_type))  # pylint: disable=too-many-function-args
  np.testing.assert_equal(argmin_f32, argmin_float_type)


def test_argmin_on_nan(float_type):
  """Ensures we return the right thing for multiple NaNs."""
  one_with_nans = np.array([1.0, float("nan"), float("nan")], dtype=np.float32)
  np.testing.assert_equal(
      np.argmin(one_with_nans.astype(float_type)), np.argmin(one_with_nans)
  )


def test_argmin_on_positive_infinity(float_type):
  """Ensures we return the right thing for positive infinities."""
  inf = np.array([float("inf")], dtype=np.float32)
  np.testing.assert_equal(np.argmin(inf.astype(float_type)), np.argmin(inf))


def test_dtype_from_string(float_type):
  assert np.dtype(float_type.__name__) == np.dtype(float_type)


BinaryOp = collections.namedtuple("BinaryOp", ["op"])

UNARY_UFUNCS = [
    np.negative,
    np.positive,
    np.absolute,
    np.fabs,
    np.rint,
    np.sign,
    np.conjugate,
    np.exp,
    np.exp2,
    np.expm1,
    np.log,
    np.log10,
    np.log1p,
    np.log2,
    np.sqrt,
    np.square,
    np.cbrt,
    np.reciprocal,
    np.sin,
    np.cos,
    np.tan,
    np.arcsin,
    np.arccos,
    np.arctan,
    np.sinh,
    np.cosh,
    np.tanh,
    np.arcsinh,
    np.arccosh,
    np.arctanh,
    np.deg2rad,
    np.rad2deg,
    np.floor,
    np.ceil,
    np.trunc,
]

BINARY_UFUNCS = [
    np.add,
    np.subtract,
    np.multiply,
    np.divide,
    np.logaddexp,
    np.logaddexp2,
    np.floor_divide,
    np.power,
    np.remainder,
    np.fmod,
    np.heaviside,
    np.arctan2,
    np.hypot,
    np.maximum,
    np.minimum,
    np.fmax,
    np.fmin,
    np.copysign,
]

BINARY_PREDICATE_UFUNCS = [
    np.equal,
    np.not_equal,
    np.less,
    np.greater,
    np.less_equal,
    np.greater_equal,
    np.logical_and,
    np.logical_or,
    np.logical_xor,
]


"""Tests NumPy integration of the custom float types."""


def test_dtype(float_type):
  assert float_type == np.dtype(float_type)


def test_deep_copy_does_not_alter_hash(float_type):
  # For context, see https://github.com/google/jax/issues/4651. If the hash
  # value of the type descriptor is not initialized correctly, a deep copy
  # can change the type hash.
  dtype = np.dtype(float_type)
  h = hash(dtype)
  _ = copy.deepcopy(dtype)
  assert h == hash(dtype)


def test_array(float_type):
  x = np.array([[1, 2, 3]], dtype=float_type)
  assert float_type == x.dtype
  assert "[[1 2 3]]" == str(x)
  np.testing.assert_equal(x, x)
  numpy_assert_allclose(x, x, float_type=float_type)
  assert (x == x).all()


def test_comparisons(float_type):
  x = np.array([30, 7, -30], dtype=np.float32)
  bx = x.astype(float_type)
  y = np.array([17, 7, 0], dtype=np.float32)
  by = y.astype(float_type)
  np.testing.assert_equal(x == y, bx == by)
  np.testing.assert_equal(x != y, bx != by)
  np.testing.assert_equal(x < y, bx < by)
  np.testing.assert_equal(x > y, bx > by)
  np.testing.assert_equal(x <= y, bx <= by)
  np.testing.assert_equal(x >= y, bx >= by)


def test_equal2(float_type):
  a = np.array([31], float_type)
  b = np.array([15], float_type)
  assert not a.__eq__(b)


def test_can_cast(float_type):
  allowed_casts = [
      (np.bool_, float_type),
      (np.int8, float_type),
      (np.uint8, float_type),
      (float_type, np.float32),
      (float_type, np.float64),
      (float_type, np.longdouble),
      (float_type, np.complex64),
      (float_type, np.complex128),
      (float_type, np.clongdouble),
  ]
  all_dtypes = [
      np.float16,
      np.float32,
      np.float64,
      np.longdouble,
      np.int8,
      np.int16,
      np.int32,
      np.int64,
      np.complex64,
      np.complex128,
      np.clongdouble,
      np.uint8,
      np.uint16,
      np.uint32,
      np.uint64,
      np.intc,
      np.int_,
      np.longlong,
      np.uintc,
      np.ulonglong,
  ]
  for d in all_dtypes:
    assert ((float_type, d) in allowed_casts) == np.can_cast(float_type, d)
    assert ((d, float_type) in allowed_casts) == np.can_cast(d, float_type)


@ignore_warning(
    category=RuntimeWarning, message="invalid value encountered in cast"
)
def test_casts(float_type):
  for dtype in [
      np.float16,
      np.float32,
      np.float64,
      np.longdouble,
      np.int8,
      np.int16,
      np.int32,
      np.int64,
      np.complex64,
      np.complex128,
      np.clongdouble,
      np.uint8,
      np.uint16,
      np.uint32,
      np.uint64,
      np.intc,
      np.int_,
      np.longlong,
      np.uintc,
      np.ulonglong,
  ]:
    x = np.array([[1, 2, 3]], dtype=dtype)
    y = x.astype(float_type)
    z = y.astype(dtype)
    assert np.all(x == y)
    assert float_type == y.dtype
    assert np.all(x == z)
    assert dtype == z.dtype


@ignore_warning(category=np.ComplexWarning)
def test_conform_numpy_complex(float_type):
  for dtype in [np.complex64, np.complex128, np.clongdouble]:
    x = np.array([1.5, 2.5 + 2.0j, 3.5], dtype=dtype)
    y_np = x.astype(np.float32)
    y_tf = x.astype(float_type)
    numpy_assert_allclose(y_np, y_tf, atol=2e-2, float_type=float_type)

    z_np = y_np.astype(dtype)
    z_tf = y_tf.astype(dtype)
    numpy_assert_allclose(z_np, z_tf, atol=2e-2, float_type=float_type)


def test_arange(float_type):
  np.testing.assert_equal(
      np.arange(100, dtype=np.float32).astype(float_type),
      np.arange(100, dtype=float_type),
  )
  np.testing.assert_equal(
      np.arange(-8, 8, 1, dtype=np.float32).astype(float_type),
      np.arange(-8, 8, 1, dtype=float_type),
  )
  np.testing.assert_equal(
      np.arange(-0.0, -2.0, -0.25, dtype=np.float32).astype(float_type),
      np.arange(-0.0, -2.0, -0.25, dtype=float_type),
  )
  np.testing.assert_equal(
      np.arange(-16.0, 16.0, 2.0, dtype=np.float32).astype(float_type),
      np.arange(-16.0, 16.0, 2.0, dtype=float_type),
  )


@ignore_warning(category=RuntimeWarning, message="invalid value encountered")
@ignore_warning(category=RuntimeWarning, message="divide by zero encountered")
def test_unary_ufunc(float_type):
  for op in UNARY_UFUNCS:
    rng = np.random.RandomState(seed=42)
    x = rng.randn(3, 7, 10).astype(float_type)
    numpy_assert_allclose(
        op(x).astype(np.float32),
        truncate(op(x.astype(np.float32)), float_type=float_type),
        rtol=1e-4,
        float_type=float_type,
    )


@ignore_warning(category=RuntimeWarning, message="invalid value encountered")
@ignore_warning(category=RuntimeWarning, message="divide by zero encountered")
def test_binary_ufunc(float_type):
  for op in BINARY_UFUNCS:
    rng = np.random.RandomState(seed=42)
    x = rng.randn(3, 7, 10).astype(float_type)
    y = rng.randn(4, 1, 7, 10).astype(float_type)
    numpy_assert_allclose(
        op(x, y).astype(np.float32),
        truncate(
            op(x.astype(np.float32), y.astype(np.float32)),
            float_type=float_type,
        ),
        rtol=1e-4,
        float_type=float_type,
    )


def test_binary_predicate_ufunc(float_type):
  for op in BINARY_PREDICATE_UFUNCS:
    rng = np.random.RandomState(seed=42)
    x = rng.randn(3, 7).astype(float_type)
    y = rng.randn(4, 1, 7).astype(float_type)
    np.testing.assert_equal(
        op(x, y), op(x.astype(np.float32), y.astype(np.float32))
    )


def test_predicate_ufunc(float_type):
  for op in [np.isfinite, np.isinf, np.isnan, np.signbit, np.logical_not]:
    rng = np.random.RandomState(seed=42)
    shape = (3, 7, 10)
    posinf_flips = rng.rand(*shape) < 0.1
    neginf_flips = rng.rand(*shape) < 0.1
    nan_flips = rng.rand(*shape) < 0.1
    vals = rng.randn(*shape)
    vals = np.where(posinf_flips, np.inf, vals)
    vals = np.where(neginf_flips, -np.inf, vals)
    vals = np.where(nan_flips, np.nan, vals)
    vals = vals.astype(float_type)
    np.testing.assert_equal(op(vals), op(vals.astype(np.float32)))


def test_div_mod(float_type):
  rng = np.random.RandomState(seed=42)
  x = rng.randn(3, 7).astype(float_type)
  y = rng.randn(4, 1, 7).astype(float_type)
  o1, o2 = np.divmod(x, y)
  e1, e2 = np.divmod(x.astype(np.float32), y.astype(np.float32))
  numpy_assert_allclose(
      o1,
      truncate(e1, float_type=float_type),
      rtol=1e-2,
      float_type=float_type,
  )
  numpy_assert_allclose(
      o2,
      truncate(e2, float_type=float_type),
      rtol=1e-2,
      float_type=float_type,
  )


def test_modf(float_type):
  rng = np.random.RandomState(seed=42)
  x = rng.randn(3, 7).astype(float_type)
  o1, o2 = np.modf(x)
  e1, e2 = np.modf(x.astype(np.float32))
  numpy_assert_allclose(
      o1.astype(np.float32),
      truncate(e1, float_type=float_type),
      rtol=1e-2,
      float_type=float_type,
  )
  numpy_assert_allclose(
      o2.astype(np.float32),
      truncate(e2, float_type=float_type),
      rtol=1e-2,
      float_type=float_type,
  )


@ignore_warning(category=RuntimeWarning, message="invalid value encountered")
def test_ldexp(float_type):
  rng = np.random.RandomState(seed=42)
  x = rng.randn(3, 7).astype(float_type)
  y = rng.randint(-50, 50, (1, 7)).astype(np.int32)
  assert np.ldexp(x, y).dtype == x.dtype
  numpy_assert_allclose(
      np.ldexp(x, y).astype(np.float32),
      truncate(np.ldexp(x.astype(np.float32), y), float_type=float_type),
      rtol=1e-2,
      atol=1e-6,
      float_type=float_type,
  )


def test_frexp(float_type):
  rng = np.random.RandomState(seed=42)
  x = rng.randn(3, 7).astype(float_type)
  mant1, exp1 = np.frexp(x)
  mant2, exp2 = np.frexp(x.astype(np.float32))
  np.testing.assert_equal(exp1, exp2)
  numpy_assert_allclose(mant1, mant2, rtol=1e-2, float_type=float_type)


def test_copysign(float_type):
  for bits in list(range(1, 128)):
    bits_type = BITS_TYPE[float_type]
    val = bits_type(bits).view(float_type)
    val_with_sign = np.copysign(val, float_type(-1))
    val_with_sign_bits = val_with_sign.view(bits_type)
    num_bits = np.iinfo(bits_type).bits
    np.testing.assert_equal(bits | (1 << (num_bits - 1)), val_with_sign_bits)


def test_next_after(float_type):
  one = np.array(1.0, dtype=float_type)
  two = np.array(2.0, dtype=float_type)
  zero = np.array(0.0, dtype=float_type)
  nan = np.array(np.nan, dtype=float_type)
  np.testing.assert_equal(np.nextafter(one, two) - one, _finfo(float_type).eps)
  np.testing.assert_equal(
      np.nextafter(one, zero) - one, -_finfo(float_type).eps / 2
  )
  np.testing.assert_equal(np.isnan(np.nextafter(nan, one)), True)
  np.testing.assert_equal(np.isnan(np.nextafter(one, nan)), True)
  np.testing.assert_equal(np.nextafter(one, one), one)
  smallest_denormal = _finfo(float_type).smallest_subnormal
  np.testing.assert_equal(np.nextafter(zero, one), smallest_denormal)
  np.testing.assert_equal(np.nextafter(zero, -one), -smallest_denormal)
  for a, b in itertools.permutations([0.0, nan], 2):
    np.testing.assert_equal(
        np.nextafter(
            np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        ),
        np.nextafter(
            np.array(a, dtype=float_type), np.array(b, dtype=float_type)
        ),
    )


@ignore_warning(category=RuntimeWarning, message="invalid value encountered")
def test_spacing(float_type):
  # Sweep a variety of binades to see that spacing gives the proper ULP.
  for i in range(
      int(np.log2(float(_finfo(float_type).smallest_subnormal))),
      int(np.log2(float(_finfo(float_type).smallest_normal))),
  ):
    power_of_two = float_type(2.0**i)
    distance = _finfo(float_type).smallest_subnormal
    np.testing.assert_equal(np.spacing(power_of_two), distance)
    np.testing.assert_equal(np.spacing(-power_of_two), -distance)
  # Normals have a distance which depends on their binade.
  for i in range(
      int(np.log2(float(_finfo(float_type).smallest_normal))),
      int(np.log2(float(_finfo(float_type).max))),
  ):
    power_of_two = float_type(2.0**i)
    distance = _finfo(float_type).eps * power_of_two
    np.testing.assert_equal(np.spacing(power_of_two), distance)
    np.testing.assert_equal(np.spacing(-power_of_two), -distance)

  # Check that spacing agrees with arithmetic involving nextafter.
  for x in FLOAT_VALUES[float_type]:
    x_float_type = float_type(x)
    spacing = np.spacing(x_float_type)
    toward = np.copysign(float_type(2.0 * np.abs(x) + 1), x_float_type)
    nextup = np.nextafter(x_float_type, toward)
    if np.isnan(spacing):
      assert np.isnan(nextup - x_float_type)
    else:
      np.testing.assert_equal(spacing, nextup - x_float_type)

  # Check that spacing for special values gives the correct answer.
  nan = float_type(float("nan"))
  np.testing.assert_equal(np.spacing(nan), np.spacing(np.float32(nan)))
  if dtype_has_inf(float_type):
    inf = float_type(float("inf"))
    np.testing.assert_equal(np.spacing(inf), np.spacing(np.float32(inf)))
