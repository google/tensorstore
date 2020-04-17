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
"""Tests of NumPy-compatible indexing operations."""

import pickle
import re

import pytest
import tensorstore as ts
import numpy as np


def test_integer():
  x = ts.IndexTransform(input_rank=3)
  assert x[1] == ts.IndexTransform(
      input_rank=2,
      output=[
          ts.OutputIndexMap(1),
          ts.OutputIndexMap(input_dimension=0),
          ts.OutputIndexMap(input_dimension=1),
      ],
  )

  assert x[..., 1] == ts.IndexTransform(
      input_rank=2,
      output=[
          ts.OutputIndexMap(input_dimension=0),
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(1),
      ],
  )

  assert x[1, 2, 3] == ts.IndexTransform(
      input_rank=0,
      output=[ts.OutputIndexMap(1),
              ts.OutputIndexMap(2),
              ts.OutputIndexMap(3)],
  )

  with pytest.raises(
      IndexError,
      match=re.escape("An index can only have a single ellipsis (`...`)"),
  ):
    x[..., 1, ...]

  with pytest.raises(
      IndexError,
      match="Indexing expression requires 4 dimensions, and cannot be applied to a domain of rank 3"
  ):
    x[1, 2, 3, 4]


def test_newaxis():
  x = ts.IndexTransform(input_rank=3)

  assert x[np.newaxis, ..., np.newaxis] == ts.IndexTransform(
      domain=[
          ts.Dim(size=1, implicit_lower=True, implicit_upper=True),
          ts.Dim(),
          ts.Dim(),
          ts.Dim(),
          ts.Dim(size=1, implicit_lower=True, implicit_upper=True),
      ],
      output=[
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(input_dimension=2),
          ts.OutputIndexMap(input_dimension=3),
      ],
  )


def test_integer_array():
  x = ts.IndexTransform(input_rank=3)

  assert x[1, [1, 2, 3]] == ts.IndexTransform(
      domain=[ts.Dim(size=3), ts.Dim()],
      output=[
          ts.OutputIndexMap(1),
          ts.OutputIndexMap(index_array=[[1], [2], [3]]),
          ts.OutputIndexMap(input_dimension=1),
      ],
  )

  assert x[[[5, 6, 7], [8, 9, 10]], [1, 2, 3]] == ts.IndexTransform(
      domain=[ts.Dim(size=2), ts.Dim(size=3),
              ts.Dim()],
      output=[
          ts.OutputIndexMap(index_array=[[[5], [6], [7]], [[8], [9], [10]]]),
          ts.OutputIndexMap(index_array=[[[1], [2], [3]]]),
          ts.OutputIndexMap(input_dimension=2),
      ],
  )

  assert x[[[5], [8]], :, [1, 2, 3]] == ts.IndexTransform(
      domain=[ts.Dim(size=2), ts.Dim(size=3),
              ts.Dim()],
      output=[
          ts.OutputIndexMap(index_array=[[[5]], [[8]]]),
          ts.OutputIndexMap(input_dimension=2),
          ts.OutputIndexMap(index_array=[[[1], [2], [3]]]),
      ],
  )

  # Test using a zero-size index array.
  assert x[[]] == ts.IndexTransform(
      domain=[ts.Dim(size=0), ts.Dim(), ts.Dim()],
      output=[
          ts.OutputIndexMap(0),
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(input_dimension=2),
      ],
  )

  with pytest.raises(
      IndexError,
      match=re.escape("Incompatible index array shapes: {3} vs {4}")):
    x[[1, 2, 3, 4], [1, 2, 3]]

  for indices in [1.5, [1.5, 2.5], "x", {"a": 3}, [None]]:
    with pytest.raises(
        IndexError,
        match=re.escape("Only integers, slices (`:`), ellipsis (`...`), "
                        "tensorstore.newaxis (`None`) and "
                        "integer or boolean arrays are valid indices"),
    ):
      x[indices]

  for indices in [np.array([1.1, 1.5]), np.array(["x", "y"])]:
    with pytest.raises(
        IndexError,
        match=re.escape(
            "Arrays used as indices must be of integer (or boolean) type"),
    ):
      x[indices]


def test_boolean_array():
  x = ts.IndexTransform(input_rank=3)

  assert x[1, [True, False, True, True]] == ts.IndexTransform(
      domain=[ts.Dim(size=3), ts.Dim()],
      output=[
          ts.OutputIndexMap(1),
          ts.OutputIndexMap(index_array=[[0], [2], [3]]),
          ts.OutputIndexMap(input_dimension=1),
      ],
  )

  assert x[[[True, False, False], [False, False, True]]] == ts.IndexTransform(
      domain=[ts.Dim(size=2), ts.Dim()],
      output=[
          ts.OutputIndexMap(index_array=[[0], [1]]),
          ts.OutputIndexMap(index_array=[[0], [2]]),
          ts.OutputIndexMap(input_dimension=1),
      ],
  )

  assert x[True] == ts.IndexTransform(
      domain=[ts.Dim(size=1), ts.Dim(),
              ts.Dim(), ts.Dim()],
      output=[
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(input_dimension=2),
          ts.OutputIndexMap(input_dimension=3),
      ],
  )


def test_slice():
  x = ts.IndexTransform(input_rank=3)

  assert x[:, 1] == ts.IndexTransform(
      input_rank=2,
      output=[
          ts.OutputIndexMap(input_dimension=0),
          ts.OutputIndexMap(1),
          ts.OutputIndexMap(input_dimension=1),
      ],
  )

  assert x[:5] == ts.IndexTransform(
      domain=[ts.Dim(exclusive_max=5),
              ts.Dim(), ts.Dim()])

  assert x[2:5] == ts.IndexTransform(
      domain=[ts.Dim(inclusive_min=2, size=3),
              ts.Dim(), ts.Dim()])

  assert x[10:1:-2] == ts.IndexTransform(
      domain=[ts.Dim(inclusive_min=-5, size=5),
              ts.Dim(), ts.Dim()],
      output=[
          ts.OutputIndexMap(stride=-2, input_dimension=0),
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(input_dimension=2),
      ],
  )

  y = ts.IndexTransform(input_shape=[5, 10])
  with pytest.raises(
      IndexError, match="Computing interval slice for dimension 0: .*"):
    y[1:6]
