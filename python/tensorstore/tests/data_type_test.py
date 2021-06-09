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
"""Tests for tensorstore.DataType."""

import pytest
import tensorstore as ts
import numpy as np


@pytest.mark.parametrize(
    "name",
    [
        "bool",
        "char",
        "byte",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "string",
        "ustring",
        "json",
    ],
)
def test_dtype_init_name(name):
  d = ts.dtype(name)
  d_attr = getattr(ts, name)
  assert d_attr.name == name
  assert d == d_attr
  assert d.name == name
  assert d.to_json() == name


@pytest.mark.parametrize(
    "name",
    [
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ],
)
def test_dtype_init_numpy_dtype(name):
  numpy_dtype = np.dtype(name)
  d = ts.dtype(numpy_dtype)
  assert d.name == name
  assert d.numpy_dtype == numpy_dtype


@pytest.mark.parametrize(
    "name",
    [
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ],
)
def test_dtype_init_type(name):
  t = getattr(np, name)
  d = ts.dtype(t)
  assert d.name == name
  assert d.type is t
  v = d(5)
  assert type(v) is t
  assert v == 5


def test_dtype_init_bool():
  assert ts.dtype(bool) == ts.bool
  assert ts.dtype(bool).type is np.bool_
  assert type(ts.bool(True)) is np.bool_
  assert ts.bool(True) == True
  assert ts.bool(False) == False


def test_dtype_init_bfloat16():
  v = ts.bfloat16(1)
  t = type(v)
  assert ts.dtype(t) == ts.bfloat16
  assert ts.bfloat16.type is t
  assert ts.bfloat16(1) == 1
  assert ts.bfloat16(1.5) == 1.5


def test_dtype_init_str():
  assert ts.dtype(str) == ts.ustring
  assert ts.dtype(str).type is str
  assert ts.ustring.numpy_dtype == np.dtype("object")
  assert ts.ustring("hello") == "hello"
  assert type(ts.ustring("hello")) is str


def test_dtype_init_bytes():
  assert ts.dtype(bytes) == ts.string
  assert ts.dtype(bytes).type is bytes
  assert ts.string.numpy_dtype == np.dtype("object")
  assert ts.string(b"hello") == b"hello"
  assert type(ts.string(b"hello")) is bytes


def assert_exact(expected, value):
  assert value == expected
  assert type(value) is type(expected)


def test_dtype_json():
  assert ts.json.numpy_dtype == np.dtype("object")
  assert ts.json((1, 2, 3)) == [1, 2, 3]
  assert ts.json.type is np.object_
  assert type(ts.json((1, 2, 3))) is list

  # Test round trip conversion between Python objects and the C++
  # `::nlohmann::json` type.
  assert_exact(ts.json(3), 3)
  assert_exact(ts.json(3.0), 3.0)
  assert_exact(ts.json(np.uint8(3)), 3)
  assert_exact(ts.json(np.uint16(3)), 3)
  assert_exact(ts.json(np.uint32(3)), 3)
  assert_exact(ts.json(np.float32(3)), 3.0)
  assert_exact(ts.json(np.float64(3)), 3.0)
  assert_exact(ts.json(np.array([1, 2, 3])), [1, 2, 3])

  for v in [
      "abc",
      {
          "x": 1,
          "y": "abc",
          "z": [1, 2, 3]
      },
  ]:
    assert_exact(ts.json(v), v)


def test_compare():
  assert ts.uint8 == np.uint8
  assert ts.bool == bool
  assert bool == ts.bool
  assert ts.bool != np.uint8
  assert np.uint8 != ts.bool
