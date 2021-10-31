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
"""Tests of tensorstore.IndexTransform"""

import pickle
import re

import pytest
import tensorstore as ts
import numpy as np


def test_identity():
  x = ts.IndexTransform(input_rank=3)
  assert x.ndim == 3
  assert x.input_rank == 3
  assert x.output_rank == 3
  assert x.input_inclusive_min == (-ts.inf,) * 3
  assert x.input_inclusive_max == (+ts.inf,) * 3
  assert x.input_exclusive_max == (+ts.inf + 1,) * 3
  assert x.input_shape == (2 * ts.inf + 1,) * 3
  assert x.output == [ts.OutputIndexMap(input_dimension=i) for i in range(3)]
  assert list(
      x.output) == [ts.OutputIndexMap(input_dimension=i) for i in range(3)]
  assert x([1, 2, 3]) == (1, 2, 3)
  with pytest.raises(ValueError):
    x([1])
  with pytest.raises(ValueError):
    x([1, 2, 3, 4])

  x = ts.IndexTransform(input_labels=['x', 'y'])
  assert x.input_labels == ('x', 'y')
  assert x.output == [ts.OutputIndexMap(input_dimension=i) for i in range(2)]


def test_init_rank_positional():
  x = ts.IndexTransform(3)
  assert x.input_rank == 3
  assert x.output_rank == 3


def test_init_domain():
  x = ts.IndexTransform(ts.IndexDomain(3))
  assert x == ts.IndexTransform(3)


def test_init_output():
  x = ts.IndexTransform(
      input_shape=[3, 2],
      output=[
          ts.OutputIndexMap(offset=7, input_dimension=1),
          ts.OutputIndexMap([[1, 2]], offset=2, stride=-1),
          ts.OutputIndexMap(8),
          ts.OutputIndexMap([[1, 2]], offset=2, stride=-1,
                            index_range=ts.Dim(inclusive_min=0,
                                               exclusive_max=8)),
      ],
  )
  assert x.output[3].index_range == ts.Dim(inclusive_min=0, exclusive_max=8)
  assert x.output == [
      ts.OutputIndexMap(offset=7, input_dimension=1),
      ts.OutputIndexMap([[1, 2]], offset=2, stride=-1),
      ts.OutputIndexMap(8),
      ts.OutputIndexMap([[1, 2]], offset=2, stride=-1,
                        index_range=ts.Dim(inclusive_min=0, exclusive_max=8)),
  ]


def test_init_output_index_maps():
  x = ts.IndexTransform(
      input_shape=[3, 2],
      output=[
          ts.OutputIndexMap(offset=7, input_dimension=1),
          ts.OutputIndexMap([[1, 2]], offset=2, stride=-1),
      ],
  )
  y = ts.IndexTransform(x.domain, x.output)
  assert x == y


def test_output_index_maps_lifetime():
  output = ts.IndexTransform(3).output
  assert output == [
      ts.OutputIndexMap(input_dimension=0),
      ts.OutputIndexMap(input_dimension=1),
      ts.OutputIndexMap(input_dimension=2),
  ]


def test_identity_error():
  with pytest.raises(ValueError):
    ts.IndexTransform(input_rank=-4)


def test_pickle():
  x = ts.IndexTransform(
      input_inclusive_min=[1, 2, -1],
      implicit_lower_bounds=[1, 0, 0],
      input_shape=[3, 2, 4],
      implicit_upper_bounds=[0, 1, 0],
      input_labels=['x', 'y', 'z'],
      output=[
          ts.OutputIndexMap(offset=7, stride=13, input_dimension=1),
          ts.OutputIndexMap(offset=8),
          ts.OutputIndexMap(
              offset=1,
              stride=-2,
              index_array=[[[-10, 1, 2, 20]]],
              index_range=ts.Dim(inclusive_min=-3, exclusive_max=10),
          ),
      ],
  )
  assert pickle.loads(pickle.dumps(x)) == x


def test_json():
  json = {
      'input_inclusive_min': [1, 2, 3],
      'input_exclusive_max': [4, 5, 6],
      'input_labels': ['x', 'y', 'z'],
      'output': [
          {
              'offset': 3
          },
          {
              'input_dimension': 0,
              'stride': 2
          },
      ],
  }
  x = ts.IndexTransform(json=json)
  expected = ts.IndexTransform(
      input_inclusive_min=[1, 2, 3],
      input_exclusive_max=[4, 5, 6],
      input_labels=['x', 'y', 'z'],
      output=[
          ts.OutputIndexMap(offset=3),
          ts.OutputIndexMap(stride=2, input_dimension=0),
      ],
  )
  assert x == expected
  assert x.to_json() == json


def test_eq():
  x = ts.IndexTransform(input_rank=2)
  y = ts.IndexTransform(input_rank=3)
  assert x == x
  assert x != y


def test_domain_access():
  x = ts.IndexTransform(input_inclusive_min=[1, 2, 3], input_shape=[5, 6, 7])
  assert x.origin == (1, 2, 3)
  assert x.shape == (5, 6, 7)
  assert x.size == 5 * 6 * 7
  assert x.T == ts.IndexTransform(
      input_inclusive_min=[3, 2, 1],
      input_shape=[7, 6, 5],
      output=[ts.OutputIndexMap(input_dimension=i) for i in [2, 1, 0]],
  )
  assert x.T == x[ts.d[::-1].transpose[:]]
