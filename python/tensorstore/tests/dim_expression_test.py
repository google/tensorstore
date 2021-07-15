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
"""Tests of tensorstore.DimExpression"""

import pickle
import re

import pytest
import tensorstore as ts
import numpy as np


def test_dimension_selection():
  assert repr(ts.d[2]) == "d[2]"
  assert repr(ts.d[2, 3]) == "d[2,3]"
  assert repr(ts.d[2, 3, "x"]) == "d[2,3,'x']"
  assert repr(ts.d[1:3, "x"]) == "d[1:3,'x']"
  assert repr(ts.d["x", "y"]) == "d['x','y']"
  assert ts.d[1] == ts.d[ts.d[1]]
  assert ts.d[1] != ts.d[2]
  assert ts.d[1, 2, 3] == ts.d[1, [2, 3]]
  assert ts.d[1, 2, 3] == ts.d[1, ts.d[2, 3]]

  with pytest.raises(TypeError):
    iter(ts.d)


def test_no_operations():
  x = ts.IndexTransform(input_rank=3)
  expr = ts.d[0, 1]
  with pytest.raises(
      IndexError,
      match="Must specify at least one operation in dimension expression"):
    x[expr]  # pylint: disable=pointless-statement


def test_translate_by_vector():
  x = ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"])

  expr = ts.d["x", "y"].translate_by[4, 5]
  assert repr(expr) == "d['x','y'].translate_by[4,5]"
  assert x[expr] == ts.IndexTransform(
      input_shape=[2, 3],
      input_inclusive_min=[4, 5],
      input_labels=["x", "y"],
      output=[
          ts.OutputIndexMap(offset=-4, input_dimension=0),
          ts.OutputIndexMap(offset=-5, input_dimension=1),
      ],
  )


def test_translate_by_scalar():
  x = ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"])

  expr = ts.d["x", "y"].translate_by[4]
  assert repr(expr) == "d['x','y'].translate_by[4]"
  assert x[expr] == ts.IndexTransform(
      input_shape=[2, 3],
      input_inclusive_min=[4, 4],
      input_labels=["x", "y"],
      output=[
          ts.OutputIndexMap(offset=-4, input_dimension=0),
          ts.OutputIndexMap(offset=-4, input_dimension=1),
      ],
  )


def test_translate_to():
  x = ts.IndexTransform(
      input_shape=[2, 3], input_inclusive_min=[1, 2], input_labels=["x", "y"])

  expr = ts.d["x", "y"].translate_to[4, 5]
  assert repr(expr) == "d['x','y'].translate_to[4,5]"
  assert x[expr] == ts.IndexTransform(
      input_shape=[2, 3],
      input_inclusive_min=[4, 5],
      input_labels=["x", "y"],
      output=[
          ts.OutputIndexMap(offset=-3, input_dimension=0),
          ts.OutputIndexMap(offset=-3, input_dimension=1),
      ],
  )


def test_stride_vector():
  x = ts.IndexTransform(
      input_inclusive_min=[0, 2, 1],
      input_inclusive_max=[6, 5, 8],
      input_labels=["x", "y", "z"])

  expr = ts.d["x", "z"].stride[-2, 3]
  assert repr(expr) == "d['x','z'].stride[-2,3]"
  assert x[expr] == ts.IndexTransform(
      input_inclusive_min=[-3, 2, 1],
      input_inclusive_max=[0, 5, 2],
      input_labels=["x", "y", "z"],
      output=[
          ts.OutputIndexMap(stride=-2, input_dimension=0),
          ts.OutputIndexMap(stride=1, input_dimension=1),
          ts.OutputIndexMap(stride=3, input_dimension=2),
      ],
  )


def test_stride_scalar():
  x = ts.IndexTransform(
      input_inclusive_min=[0, 2, 1],
      input_inclusive_max=[6, 5, 8],
      input_labels=["x", "y", "z"])

  expr = ts.d["x", "z"].stride[3]
  assert repr(expr) == "d['x','z'].stride[3]"
  assert x[expr] == ts.IndexTransform(
      input_inclusive_min=[0, 2, 1],
      input_inclusive_max=[2, 5, 2],
      input_labels=["x", "y", "z"],
      output=[
          ts.OutputIndexMap(stride=3, input_dimension=0),
          ts.OutputIndexMap(stride=1, input_dimension=1),
          ts.OutputIndexMap(stride=3, input_dimension=2),
      ],
  )


def test_label_single():
  x = ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"])

  expr = ts.d["x"].label["a"]
  assert repr(expr) == "d['x'].label['a']"
  assert x[expr] == ts.IndexTransform(
      input_shape=[2, 3], input_labels=["a", "y"])


def test_label_multiple():
  x = ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"])

  expr = ts.d["x", "y"].label["a", "b"]
  assert repr(expr) == "d['x','y'].label['a','b']"
  assert x[expr] == ts.IndexTransform(
      input_shape=[2, 3], input_labels=["a", "b"])


def test_label_vector():
  x = ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"])

  expr = ts.d["x", "y"].label[["a", "b"]]
  assert repr(expr) == "d['x','y'].label['a','b']"
  assert x[expr] == ts.IndexTransform(
      input_shape=[2, 3], input_labels=["a", "b"])


def test_label_wrong_number():
  transform = ts.IndexTransform(3)
  with pytest.raises(IndexError):
    transform[ts.d[0].label["x", "y"]]


def test_label_duplicates():
  transform = ts.IndexTransform(3)[ts.d[0].label["x"]]
  with pytest.raises(IndexError):
    transform[ts.d[1].label["x"]]


def test_add_new():
  x = ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"])
  expr = ts.d[0, -2:][ts.newaxis]
  assert repr(expr) == "d[0,-2:][None]"
  assert x[expr] == ts.IndexTransform(
      input_inclusive_min=[0, 0, 0, 0, 0],
      input_exclusive_max=[1, 2, 3, 1, 1],
      input_labels=["", "x", "y", "", ""],
      implicit_lower_bounds=[1, 0, 0, 1, 1],
      implicit_upper_bounds=[1, 0, 0, 1, 1],
      output=[
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(input_dimension=2),
      ],
  )


def test_add_new_invalid_rank():
  x = ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"])
  expr = ts.d[0:32][ts.newaxis]
  with pytest.raises(IndexError):
    x[expr]


def test_diagonal():
  x = ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"])
  expr = ts.d["x", "y"].diagonal
  assert repr(expr) == "d['x','y'].diagonal"
  assert x[expr] == ts.IndexTransform(
      input_shape=[2],
      output=[
          ts.OutputIndexMap(input_dimension=0),
          ts.OutputIndexMap(input_dimension=0),
      ],
  )


def test_transpose_dim_range():
  x = ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"])
  expr = ts.d["y", "x"].transpose[:]
  assert repr(expr) == "d['y','x'].transpose[:]"
  assert x[expr] == ts.IndexTransform(
      input_shape=[3, 2],
      input_labels=["y", "x"],
      output=[
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(input_dimension=0),
      ],
  )


def test_transpose_single_dim():
  x = ts.IndexTransform(input_labels=["x", "y", "z"])
  expr = ts.d["x"].transpose[1]
  assert repr(expr) == "d['x'].transpose[1]"
  assert x[expr] == ts.IndexTransform(
      input_labels=["y", "x", "z"],
      output=[
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(input_dimension=0),
          ts.OutputIndexMap(input_dimension=2),
      ],
  )


def test_transpose_move_to_front():
  x = ts.IndexTransform(input_labels=["x", "y", "z"])
  expr = ts.d["y", "x"].transpose[0]
  assert repr(expr) == "d['y','x'].transpose[0]"
  assert x[expr] == ts.IndexTransform(
      input_labels=["y", "x", "z"],
      output=[
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(input_dimension=0),
          ts.OutputIndexMap(input_dimension=2),
      ],
  )


def test_transpose_move_to_front_with_indices():
  x = ts.IndexTransform(input_labels=["x", "y", "z"])
  expr = ts.d["y", "x"].transpose[0, 1]
  assert repr(expr) == "d['y','x'].transpose[0,1]"
  assert x[expr] == ts.IndexTransform(
      input_labels=["y", "x", "z"],
      output=[
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(input_dimension=0),
          ts.OutputIndexMap(input_dimension=2),
      ],
  )


def test_transpose_empty():
  x = ts.IndexTransform(input_labels=["x", "y", "z"])
  expr = ts.d[()].transpose[()]
  assert repr(expr) == "d[()].transpose[()]"
  assert x[expr] == x


def test_transpose_label_target():
  x = ts.IndexTransform(input_labels=["x", "y", "z"])
  with pytest.raises(IndexError,
                     match="Target dimensions cannot be specified by label"):
    x[ts.d["x", "y"].transpose["x"]]


def test_transpose_move_to_back():
  x = ts.IndexTransform(input_labels=["x", "y", "z"])
  expr = ts.d["y", "x"].transpose[-1]
  assert repr(expr) == "d['y','x'].transpose[-1]"
  assert x[expr] == ts.IndexTransform(
      input_labels=["z", "y", "x"],
      output=[
          ts.OutputIndexMap(input_dimension=2),
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(input_dimension=0),
      ],
  )


def test_index_integer():
  x = ts.IndexTransform(input_shape=[15, 20], input_labels=["x", "y"])
  expr = ts.d["x", "y"][2, 3]
  assert repr(expr) == "d['x','y'][2,3]"
  assert x[expr] == ts.IndexTransform(
      input_rank=0,
      output=[ts.OutputIndexMap(offset=2),
              ts.OutputIndexMap(offset=3)],
  )


def test_index_integer_non_scalar():
  x = ts.IndexTransform(input_shape=[15, 20], input_labels=["x", "y"])
  expr = ts.d["x"][2,]
  assert repr(expr) == "d['x'][2,]"
  assert x[expr] == ts.IndexTransform(
      domain=x.domain["y",],
      output=[
          ts.OutputIndexMap(offset=2),
          ts.OutputIndexMap(input_dimension=0)
      ],
  )


def test_index_slice():
  x = ts.IndexTransform(input_shape=[15, 20], input_labels=["x", "y"])
  expr = ts.d["x", "y"][(1, 2):(7, 8)]
  assert repr(expr) == "d['x','y'][1:7,2:8]"
  assert x[expr] == ts.IndexTransform(
      input_inclusive_min=[1, 2],
      input_exclusive_max=[7, 8],
      input_labels=["x", "y"],
  )


def test_index_slice_incompatible_stop():
  with pytest.raises(
      IndexError,
      match=re.escape(
          "stop=[7,8,9] (rank 3) is incompatible with start=[1,2] (rank 2)"),
  ):
    ts.d[:][(1, 2):(7, 8, 9)]


def test_index_slice_incompatible_step():
  with pytest.raises(
      IndexError,
      match=re.escape(
          "step=[9,10,11] (rank 3) is incompatible with stop=[7,8] (rank 2)"),
  ):
    ts.d[:][(1, 2):(7, 8):(9, 10, 11)]


def test_index_slice_invalid_start():
  with pytest.raises(
      TypeError,
      match=re.escape(
          "slice indices must be integers or None or have an __index__ method"),
  ):
    ts.d[:]["a":3]


def test_index_slice_invalid_stop():
  with pytest.raises(TypeError):
    ts.d[:][3:"a"]


def test_index_slice_invalid_step():
  with pytest.raises(TypeError):
    ts.d[:][3:5:"a"]


def test_label_index_slice():
  x = ts.IndexTransform(input_labels=["x", "y"])
  expr = ts.d["x", "y"].label["a", "b"][(1, 2):(7, 8)]
  assert repr(expr) == "d['x','y'].label['a','b'][1:7,2:8]"
  assert x[expr] == ts.IndexTransform(
      input_inclusive_min=[1, 2],
      input_exclusive_max=[7, 8],
      input_labels=["a", "b"],
  )


def test_slice_interval_strided():
  x = ts.IndexTransform(input_shape=[15, 20], input_labels=["x", "y"])
  expr = ts.d["x", "y"][(1, 2):(8, 9):2]
  assert repr(expr) == "d['x','y'][1:8:2,2:9:2]"
  assert x[expr] == ts.IndexTransform(
      input_inclusive_min=[0, 1],
      input_inclusive_max=[3, 4],
      input_labels=["x", "y"],
      output=[
          ts.OutputIndexMap(input_dimension=0, offset=1, stride=2),
          ts.OutputIndexMap(input_dimension=1, stride=2),
      ],
  )


def test_index_repr():
  expr = ts.d[:][ts.newaxis, ..., 3, 4:5, [[False, True], [True, False]],
                 np.array([1, 2])]
  assert repr(expr) == (
      "d[:][None,...,3,4:5," +
      repr(np.array([[False, True], [True, False]], dtype=bool)) + "," +
      repr(np.array([1, 2], dtype=np.int64)) + "]")


def test_index_too_many_ops():
  x = ts.IndexTransform(input_rank=2)
  with pytest.raises(
      IndexError,
      match=re.escape(
          "Indexing expression requires 3 dimensions, and cannot be applied to a domain of rank 2"
      )):
    x[1, 2, 3]


def test_dimension_selection_index_too_many_ops():
  x = ts.IndexTransform(input_rank=2)
  with pytest.raises(
      IndexError,
      match=re.escape(
          "Indexing expression requires 3 dimensions but selection has 2 dimensions"
      ),
  ):
    x[ts.d[0, 1][1, 2, 3]]


def test_dimension_selection_chained_newaxis():
  x = ts.IndexTransform(input_rank=2)
  with pytest.raises(
      IndexError,
      match=re.escape(
          "tensorstore.newaxis (`None`) not valid in chained indexing operations"
      ),
  ):
    x[ts.d[:].label["x", "y"][ts.newaxis, ...]]


def test_dimension_selection_index_arrays_non_consecutive():
  x = ts.IndexTransform(input_labels=["a", "b", "c", "d"])
  expr = ts.d["a", "c", "d"][[[1, 2, 3], [4, 5, 6]], :, [6, 7, 8]]
  assert x[expr] == ts.IndexTransform(
      domain=[
          ts.Dim(size=2),
          ts.Dim(size=3),
          ts.Dim(label="b"),
          ts.Dim(label="c"),
      ],
      output=[
          ts.OutputIndexMap(
              index_array=[[[[1]], [[2]], [[3]]], [[[4]], [[5]], [[6]]]]),
          ts.OutputIndexMap(input_dimension=2),
          ts.OutputIndexMap(input_dimension=3),
          ts.OutputIndexMap(index_array=[[[[6]], [[7]], [[8]]]]),
      ],
  )


def test_dimension_selection_index_arrays_consecutive():
  x = ts.IndexTransform(input_labels=["a", "b", "c", "d"])
  expr = ts.d["a", "c", "d"][..., [[1, 2, 3], [4, 5, 6]], [6, 7, 8]]
  assert x[expr] == ts.IndexTransform(
      domain=[
          ts.Dim(label="a"),
          ts.Dim(label="b"),
          ts.Dim(size=2),
          ts.Dim(size=3),
      ],
      output=[
          ts.OutputIndexMap(input_dimension=0),
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(index_array=[[[[1, 2, 3], [4, 5, 6]]]]),
          ts.OutputIndexMap(index_array=[[[[6, 7, 8]]]]),
      ],
  )


def test_dimension_selection_bool_arrays_non_consecutive():
  x = ts.IndexTransform(input_labels=["a", "b", "c", "d"])
  expr = ts.d["a", "b", "c",
              "d"][[[False, True, True], [False, False, False]], :,
                   [False, False, True, True]]
  assert x[expr] == ts.IndexTransform(
      domain=[ts.Dim(size=2), ts.Dim(label="c")],
      output=[
          ts.OutputIndexMap(index_array=[[0], [0]]),
          ts.OutputIndexMap(index_array=[[1], [2]]),
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(index_array=[[2], [3]]),
      ],
  )


def test_dimension_selection_bool_arrays_consecutive():
  x = ts.IndexTransform(input_labels=["a", "b", "c", "d"])
  expr = ts.d["a", "b", "c",
              "d"][:, [[False, True, True], [False, False, False]],
                   [False, False, True, True]]
  assert x[expr] == ts.IndexTransform(
      domain=[ts.Dim(label="a"), ts.Dim(size=2)],
      output=[
          ts.OutputIndexMap(input_dimension=0),
          ts.OutputIndexMap(index_array=[[0, 0]]),
          ts.OutputIndexMap(index_array=[[1, 2]]),
          ts.OutputIndexMap(index_array=[[2, 3]]),
      ],
  )


def test_dimension_selection_oindex_bool_arrays():
  x = ts.IndexTransform(input_labels=["a", "b", "c", "d"])
  expr = ts.d["a", "b", "c",
              "d"].oindex[[[False, True, True], [False, False, False]], :,
                          [True, False, True, True]]
  assert x[expr] == ts.IndexTransform(
      domain=[ts.Dim(size=2), ts.Dim(label="c"),
              ts.Dim(size=3)],
      output=[
          ts.OutputIndexMap(index_array=[[[0]], [[0]]]),
          ts.OutputIndexMap(index_array=[[[1]], [[2]]]),
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(index_array=[[[0, 2, 3]]]),
      ],
  )


def test_dimension_selection_vindex_bool_arrays_non_consecutive():
  x = ts.IndexTransform(input_labels=["a", "b", "c", "d"])
  expr = ts.d["a", "b", "c",
              "d"].vindex[[[False, True, True], [False, False, False]], :,
                          [False, False, True, True]]
  assert x[expr] == ts.IndexTransform(
      domain=[ts.Dim(size=2), ts.Dim(label="c")],
      output=[
          ts.OutputIndexMap(index_array=[[0], [0]]),
          ts.OutputIndexMap(index_array=[[1], [2]]),
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(index_array=[[2], [3]]),
      ],
  )


def test_dimension_selection_vindex_repr():
  assert repr(ts.d[:].vindex[1, 2]) == "d[:].vindex[1,2]"


def test_dimension_selection_oindex_repr():
  assert repr(ts.d[:].oindex[1, 2]) == "d[:].oindex[1,2]"


def test_dimension_selection_index_repr():
  assert repr(ts.d[:][1, 2]) == "d[:][1,2]"


def test_dimension_selection_vindex_bool_arrays_consecutive():
  x = ts.IndexTransform(input_labels=["a", "b", "c", "d"])
  expr = ts.d["a", "b", "c",
              "d"].vindex[:, [[False, True, True], [False, False, False]],
                          [False, False, True, True]]
  assert x[expr] == ts.IndexTransform(
      domain=[ts.Dim(size=2), ts.Dim(label="a")],
      output=[
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(index_array=[[0], [0]]),
          ts.OutputIndexMap(index_array=[[1], [2]]),
          ts.OutputIndexMap(index_array=[[2], [3]]),
      ],
  )


def test_dimension_selection_oindex_index_arrays():
  x = ts.IndexTransform(input_labels=["a", "b", "c", "d"])
  expr = ts.d["a", "c", "d"].oindex[[[1, 2, 3], [4, 5, 6]], :, [6, 7, 8]]
  assert x[expr] == ts.IndexTransform(
      domain=[
          ts.Dim(size=2),
          ts.Dim(size=3),
          ts.Dim(label="b"),
          ts.Dim(label="c"),
          ts.Dim(size=3),
      ],
      output=[
          ts.OutputIndexMap(index_array=[[[[[1]]], [[[2]]], [[[3]]]],
                                         [[[[4]]], [[[5]]], [[[6]]]]]),
          ts.OutputIndexMap(input_dimension=2),
          ts.OutputIndexMap(input_dimension=3),
          ts.OutputIndexMap(index_array=[[[[[6, 7, 8]]]]]),
      ],
  )


def test_dimension_selection_vindex_index_arrays_non_consecutive():
  x = ts.IndexTransform(input_labels=["a", "b", "c", "d"])
  expr = ts.d["a", "c", "d"].vindex[[[1, 2, 3], [4, 5, 6]], :, [6, 7, 8]]
  assert x[expr] == ts.IndexTransform(
      domain=[
          ts.Dim(size=2),
          ts.Dim(size=3),
          ts.Dim(label="b"),
          ts.Dim(label="c"),
      ],
      output=[
          ts.OutputIndexMap(
              index_array=[[[[1]], [[2]], [[3]]], [[[4]], [[5]], [[6]]]]),
          ts.OutputIndexMap(input_dimension=2),
          ts.OutputIndexMap(input_dimension=3),
          ts.OutputIndexMap(index_array=[[[[6]], [[7]], [[8]]]]),
      ],
  )


def test_dimension_selection_vindex_index_arrays_consecutive():
  x = ts.IndexTransform(input_labels=["a", "b", "c", "d"])
  expr = ts.d["a", "c", "d"].vindex[..., [[1, 2, 3], [4, 5, 6]], [6, 7, 8]]
  assert x[expr] == ts.IndexTransform(
      domain=[
          ts.Dim(size=2),
          ts.Dim(size=3),
          ts.Dim(label="a"),
          ts.Dim(label="b"),
      ],
      output=[
          ts.OutputIndexMap(input_dimension=2),
          ts.OutputIndexMap(input_dimension=3),
          ts.OutputIndexMap(
              index_array=[[[[1]], [[2]], [[3]]], [[[4]], [[5]], [[6]]]]),
          ts.OutputIndexMap(index_array=[[[[6]], [[7]], [[8]]]]),
      ],
  )


def test_dimension_selection_dimrange_index():
  x = ts.IndexTransform(input_labels=["a", "b", "c"])
  assert x[ts.d["c", :2][:5, 1, 2]] == ts.IndexTransform(
      input_labels=["c"],
      input_exclusive_max=[5],
      output=[
          ts.OutputIndexMap(1),
          ts.OutputIndexMap(2),
          ts.OutputIndexMap(input_dimension=0),
      ],
  )


def test_dimension_selection_dimrange_index_invalid_start():
  x = ts.IndexTransform(input_rank=3)
  expr = ts.d[5:][...]
  with pytest.raises(
      IndexError,
      match=re.escape("Dimension index 5 is outside valid range [-3, 3)"),
  ):
    x[expr]


def test_dimension_selection_dimrange_index_invalid_stop():
  x = ts.IndexTransform(input_rank=3)
  expr = ts.d[:5][...]
  with pytest.raises(
      IndexError,
      match=re.escape(
          "Dimension exclusive stop index 5 is outside valid range [-4, 3]"),
  ):
    x[expr]


def test_dimension_selection_duplicate_index():
  x = ts.IndexTransform(input_rank=3)
  expr = ts.d[1, 1][...]
  with pytest.raises(
      IndexError, match=re.escape("Dimension 1 specified more than once")):
    x[expr]


def test_dimension_selection_duplicate_index_label():
  x = ts.IndexTransform(input_labels=["x", "y", "z"])
  expr = ts.d[1, "y"][...]
  with pytest.raises(
      IndexError, match=re.escape("Dimension 1 specified more than once")):
    x[expr]


def test_dimension_selection_duplicate_index_label_newaxis():
  x = ts.IndexTransform(input_labels=["x", "y", "z"])
  expr = ts.d[0, 2, "y"][ts.newaxis, ...]
  with pytest.raises(
      IndexError, match=re.escape("Dimension 2 specified more than once")):
    x[expr]


def test_dimension_selection_index_label_newaxis():
  x = ts.IndexTransform(input_labels=["x", "y", "z"])
  expr = ts.d[0, "y"][ts.newaxis, ts.newaxis]
  with pytest.raises(
      IndexError,
      match=re.escape(
          "Dimensions specified by label cannot be used with newaxis"),
  ):
    x[expr]


def test_dimension_selection_index_zero_rank_bool():
  x = ts.IndexTransform(input_rank=2)
  assert x[ts.d[:][..., True]] == ts.IndexTransform(
      domain=[ts.Dim(size=1), *x.domain],
      output=[
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(input_dimension=2),
      ],
  )


def test_dimension_selection_oindex_zero_rank_bool():
  x = ts.IndexTransform(input_rank=2)
  with pytest.raises(
      IndexError,
      match=re.escape(
          "Zero-rank bool array incompatible with outer indexing of a dimension selection"
      ),
  ):
    x[ts.d[:].oindex[..., True]]
