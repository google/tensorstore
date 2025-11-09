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
"""Tests of tensorstore.Spec"""

import pickle

import pytest
import numpy as np
import tensorstore as ts


def test_spec_init_json():
  s = ts.Spec({
      "driver": "array",
      "array": [[1, 2], [3, 4]],
      "transform": {
          "input_rank": 2
      },
      "dtype": "int32",
  })
  assert s.transform == ts.IndexTransform(input_rank=2)
  assert s.rank == 2
  assert s.ndim == 2
  assert s.dtype == ts.int32
  assert s.to_json(include_defaults=False) == {
      "driver": "array",
      "array": [[1, 2], [3, 4]],
      "dtype": "int32",
      "transform": {
          "input_rank": 2
      },
  }
  assert s.T == ts.Spec({
      "driver": "array",
      "array": [[1, 2], [3, 4]],
      "transform": {
          "input_rank": 2,
          "output": [
              {
                  "input_dimension": 1
              },
              {
                  "input_dimension": 0
              },
          ],
      },
      "dtype": "int32",
  })


def test_spec_pickle():
  driver_json = {"$type": "array", "array": [[1, 2], [3, 4]]}
  s = ts.Spec({
      "driver": "array",
      "array": [[1, 2], [3, 4]],
      "dtype": "int32",
      "transform": {
          "input_inclusive_min": [["-inf"], ["-inf"]],
          "input_exclusive_max": [["+inf"], ["+inf"]],
      },
  })
  assert pickle.loads(pickle.dumps(s)) == s


def test_spec_indexing():
  transform = ts.IndexTransform(input_rank=2)
  s = ts.Spec({
      "driver": "array",
      "array": [[1, 2], [3, 4]],
      "dtype": "int32",
      "transform": {
          "input_inclusive_min": [["-inf"], ["-inf"]],
          "input_exclusive_max": [["+inf"], ["+inf"]],
      },
  })
  s_transformed = s[..., ts.newaxis]
  s_expected = ts.Spec({
      "driver": "array",
      "array": [[1, 2], [3, 4]],
      "dtype": "int32",
      "transform": transform[..., ts.newaxis].to_json(),
  })
  assert s_transformed == s_expected


def test_spec_indexing_unknown_rank():
  s = ts.Spec({
      "driver": "zarr",
      "kvstore": {
          "driver": "memory"
      },
      "dtype": "int32",
  })
  assert s.rank is None
  assert s.ndim is None
  with pytest.raises(
      ValueError,
      match="Cannot perform indexing operations on Spec with unspecified rank"):
    s[..., ts.newaxis]
  with pytest.raises(
      ValueError,
      match="Cannot perform indexing operations on Spec with unspecified rank"):
    s.T


def test_codec_spec():
  s = ts.CodecSpec({"driver": "zarr", "compressor": None})
  assert s.to_json() == {"driver": "zarr", "compressor": None}


def test_schema_from_json():
  s = ts.Schema({
      "dtype": "int32",
      "chunk_layout": {
          "read_chunk": {
              "elements": 10000
          }
      }
  })
  assert s.to_json() == {
      "dtype": "int32",
      "chunk_layout": {
          "read_chunk": {
              "elements": 10000
          }
      }
  }


def test_schema():
  s = ts.Schema(dtype=ts.int32, fill_value=42)
  assert s.to_json() == {"dtype": "int32", "fill_value": 42}
  s.update(chunk_layout=ts.ChunkLayout(read_chunk_elements=5))
  assert s.to_json() == {
      "dtype": "int32",
      "fill_value": 42,
      "chunk_layout": {
          "read_chunk": {
              "elements": 5
          }
      },
  }


def test_schema_pickle():
  s = ts.Schema(dtype=ts.int32, fill_value=42)
  assert s.dtype == ts.int32
  assert pickle.loads(pickle.dumps(s)) == s
