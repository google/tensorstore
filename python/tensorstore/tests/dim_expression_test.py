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
"""Tests of tensorstore.DimExpression."""

import pickle
import re
from typing import List, Optional

import numpy as np
import pytest
import tensorstore as ts


def check_expr(expr: ts.DimExpression, expected_repr: Optional[str] = None,
               before: Optional[ts.IndexTransform] = None,
               after: Optional[ts.IndexTransform] = None):
  if expected_repr is not None:
    assert repr(expr) == expected_repr

  assert expr == expr  # pylint: disable=comparison-with-itself

  expr_repr = repr(expr)

  # Check that `expr` round trips through `repr`.
  eval_globals = {"d": ts.d, "array": np.array, 'int64': np.int64}
  # pylint: disable-next=eval-used
  assert eval(expr_repr, eval_globals) == expr

  # Check that `expr` round trips through pickling.
  unpickled = pickle.loads(pickle.dumps(expr))
  assert unpickled == expr

  if before is not None:
    assert after is not None

    assert before[expr] == after
    assert before[unpickled] == after

    # Check that expression has same result when applied to `IndexDomain`.
    assert before.domain[expr] == after.domain


def check_expr_equality(exprs: List[ts.DimExpression]):
  for a in exprs:
    for b in exprs:
      if a is b:
        assert a == a  # pylint: disable=comparison-with-itself
      else:
        assert a != b


def test_dimension_selection():
  check_expr(
      expr=ts.d[2],
      expected_repr="d[2]",
  )
  check_expr(
      expr=ts.d[2, 3],
      expected_repr="d[2,3]",
  )
  check_expr(
      expr=ts.d[2, 3, "x"],
      expected_repr="d[2,3,'x']",
  )
  check_expr(
      expr=ts.d[1:3, "x"],
      expected_repr="d[1:3,'x']",
  )
  check_expr(
      expr=ts.d["x", "y"],
      expected_repr="d['x','y']",
  )
  assert ts.d[1] == ts.d[ts.d[1]]
  assert ts.d[1] != ts.d[2]
  assert ts.d[1, 2, 3] == ts.d[1, [2, 3]]
  assert ts.d[1, 2, 3] == ts.d[1, ts.d[2, 3]]
  check_expr_equality([
      ts.d[:],
      ts.d[1],
      ts.d[2],
      ts.d["x"],
      ts.d["y"],
      ts.d[1:3],
      ts.d[1:4],
      ts.d[1:4:2],
  ])

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
  check_expr(
      expr=ts.d["x", "y"].translate_by[4, 5],
      expected_repr="d['x','y'].translate_by[4,5]",
      before=ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"]),
      after=ts.IndexTransform(
          input_shape=[2, 3],
          input_inclusive_min=[4, 5],
          input_labels=["x", "y"],
          output=[
              ts.OutputIndexMap(offset=-4, input_dimension=0),
              ts.OutputIndexMap(offset=-5, input_dimension=1),
          ],
      ),
  )


def test_translate_by_vector_with_none():
  check_expr(
      expr=ts.d["x", "y"].translate_by[4, None],
      expected_repr="d['x','y'].translate_by[4,None]",
      before=ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"]),
      after=ts.IndexTransform(
          input_shape=[2, 3],
          input_inclusive_min=[4, 0],
          input_labels=["x", "y"],
          output=[
              ts.OutputIndexMap(offset=-4, input_dimension=0),
              ts.OutputIndexMap(input_dimension=1),
          ],
      ),
  )


def test_translate_by_scalar():
  check_expr(
      expr=ts.d["x", "y"].translate_by[4],
      before=ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"]),
      expected_repr="d['x','y'].translate_by[4]",
      after=ts.IndexTransform(
          input_shape=[2, 3],
          input_inclusive_min=[4, 4],
          input_labels=["x", "y"],
          output=[
              ts.OutputIndexMap(offset=-4, input_dimension=0),
              ts.OutputIndexMap(offset=-4, input_dimension=1),
          ],
      ),
  )


def test_translate_equality():
  check_expr_equality([
      ts.d[:].translate_to[4],
      ts.d[:].translate_by[4],
      ts.d[:].translate_backward_by[4],
      ts.d[:].translate_to[4, 5],
      ts.d[:].translate_to[4, 6],
      ts.d[:2].translate_to[4, 5],
  ])


def test_translate_to():
  check_expr(
      expr=ts.d["x", "y"].translate_to[4, 5],
      before=ts.IndexTransform(input_shape=[2, 3], input_inclusive_min=[1, 2],
                               input_labels=["x", "y"]),
      expected_repr="d['x','y'].translate_to[4,5]",
      after=ts.IndexTransform(
          input_shape=[2, 3],
          input_inclusive_min=[4, 5],
          input_labels=["x", "y"],
          output=[
              ts.OutputIndexMap(offset=-3, input_dimension=0),
              ts.OutputIndexMap(offset=-3, input_dimension=1),
          ],
      ),
  )


def test_stride_vector():
  check_expr(
      expr=ts.d["x", "z"].stride[-2, 3],
      expected_repr="d['x','z'].stride[-2,3]",
      before=ts.IndexTransform(input_inclusive_min=[0, 2, 1],
                               input_inclusive_max=[6, 5, 8],
                               input_labels=["x", "y", "z"]),
      after=ts.IndexTransform(
          input_inclusive_min=[-3, 2, 1],
          input_inclusive_max=[0, 5, 2],
          input_labels=["x", "y", "z"],
          output=[
              ts.OutputIndexMap(stride=-2, input_dimension=0),
              ts.OutputIndexMap(stride=1, input_dimension=1),
              ts.OutputIndexMap(stride=3, input_dimension=2),
          ],
      ),
  )


def test_stride_scalar():
  check_expr(
      expr=ts.d["x", "z"].stride[3],
      expected_repr="d['x','z'].stride[3]",
      before=ts.IndexTransform(input_inclusive_min=[0, 2, 1],
                               input_inclusive_max=[6, 5, 8],
                               input_labels=["x", "y", "z"]),
      after=ts.IndexTransform(
          input_inclusive_min=[0, 2, 1],
          input_inclusive_max=[2, 5, 2],
          input_labels=["x", "y", "z"],
          output=[
              ts.OutputIndexMap(stride=3, input_dimension=0),
              ts.OutputIndexMap(stride=1, input_dimension=1),
              ts.OutputIndexMap(stride=3, input_dimension=2),
          ],
      ),
  )


def test_stride_equality():
  check_expr_equality([
      ts.d[:].stride[4],
      ts.d[:].stride[5],
      ts.d[:].stride[4, 5],
      ts.d[:].stride[4, 6],
  ])


def test_label_single():
  check_expr(
      expr=ts.d["x"].label["a"],
      expected_repr="d['x'].label['a']",
      before=ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"]),
      after=ts.IndexTransform(input_shape=[2, 3], input_labels=["a", "y"]),
  )


def test_label_multiple():
  check_expr(
      expr=ts.d["x", "y"].label["a", "b"],
      expected_repr="d['x','y'].label['a','b']",
      before=ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"]),
      after=ts.IndexTransform(input_shape=[2, 3], input_labels=["a", "b"]),
  )


def test_label_vector():
  check_expr(
      expr=ts.d["x", "y"].label[["a", "b"]],
      expected_repr="d['x','y'].label['a','b']",
      before=ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"]),
      after=ts.IndexTransform(input_shape=[2, 3], input_labels=["a", "b"]),
  )


def test_label_equality():
  check_expr_equality([
      ts.d[:].label["a", "b"],
      ts.d[:].label["a", "c"],
      ts.d[:].label["a"],
      ts.d[:].label["c"],
  ])


def test_label_wrong_number():
  transform = ts.IndexTransform(3)
  with pytest.raises(IndexError):
    transform[ts.d[0].label["x", "y"]]  # pylint: disable=pointless-statement


def test_label_duplicates():
  transform = ts.IndexTransform(3)[ts.d[0].label["x"]]
  with pytest.raises(IndexError):
    transform[ts.d[1].label["x"]]  # pylint: disable=pointless-statement


def test_add_new():
  check_expr(
      expr=ts.d[0, -2:][ts.newaxis],
      expected_repr="d[0,-2:][None]",
      before=ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"]),
      after=ts.IndexTransform(
          input_inclusive_min=[0, 0, 0, 0, 0],
          input_exclusive_max=[1, 2, 3, 1, 1],
          input_labels=["", "x", "y", "", ""],
          implicit_lower_bounds=[1, 0, 0, 1, 1],
          implicit_upper_bounds=[1, 0, 0, 1, 1],
          output=[
              ts.OutputIndexMap(input_dimension=1),
              ts.OutputIndexMap(input_dimension=2),
          ],
      ),
  )


def test_add_new_invalid_rank():
  x = ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"])
  expr = ts.d[0:32][ts.newaxis]
  with pytest.raises(IndexError):
    x[expr]  # pylint: disable=pointless-statement


def test_diagonal():
  check_expr(
      expr=ts.d["x", "y"].diagonal,
      expected_repr="d['x','y'].diagonal",
      before=ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"]),
      after=ts.IndexTransform(
          input_shape=[2],
          output=[
              ts.OutputIndexMap(input_dimension=0),
              ts.OutputIndexMap(input_dimension=0),
          ],
      ),
  )


def test_transpose_dim_range():
  check_expr(
      expr=ts.d["y", "x"].transpose[:],
      expected_repr="d['y','x'].transpose[:]",
      before=ts.IndexTransform(input_shape=[2, 3], input_labels=["x", "y"]),
      after=ts.IndexTransform(
          input_shape=[3, 2],
          input_labels=["y", "x"],
          output=[
              ts.OutputIndexMap(input_dimension=1),
              ts.OutputIndexMap(input_dimension=0),
          ],
      ),
  )


def test_transpose_single_dim():
  check_expr(
      expr=ts.d["x"].transpose[1],
      expected_repr="d['x'].transpose[1]",
      before=ts.IndexTransform(input_labels=["x", "y", "z"]),
      after=ts.IndexTransform(
          input_labels=["y", "x", "z"],
          output=[
              ts.OutputIndexMap(input_dimension=1),
              ts.OutputIndexMap(input_dimension=0),
              ts.OutputIndexMap(input_dimension=2),
          ],
      ),
  )


def test_transpose_move_to_front():
  check_expr(
      expr=ts.d["y", "x"].transpose[0],
      expected_repr="d['y','x'].transpose[0]",
      before=ts.IndexTransform(input_labels=["x", "y", "z"]),
      after=ts.IndexTransform(
          input_labels=["y", "x", "z"],
          output=[
              ts.OutputIndexMap(input_dimension=1),
              ts.OutputIndexMap(input_dimension=0),
              ts.OutputIndexMap(input_dimension=2),
          ],
      ),
  )


def test_transpose_move_to_front_with_indices():
  check_expr(
      expr=ts.d["y", "x"].transpose[0, 1],
      expected_repr="d['y','x'].transpose[0,1]",
      before=ts.IndexTransform(input_labels=["x", "y", "z"]),
      after=ts.IndexTransform(
          input_labels=["y", "x", "z"],
          output=[
              ts.OutputIndexMap(input_dimension=1),
              ts.OutputIndexMap(input_dimension=0),
              ts.OutputIndexMap(input_dimension=2),
          ],
      ),
  )


def test_transpose_empty():
  x = ts.IndexTransform(input_labels=["x", "y", "z"])
  check_expr(
      expr=ts.d[()].transpose[()],
      expected_repr="d[()].transpose[()]",
      before=x,
      after=x,
  )


def test_transpose_label_target():
  x = ts.IndexTransform(input_labels=["x", "y", "z"])
  with pytest.raises(IndexError,
                     match="Target dimensions cannot be specified by label"):
    x[ts.d["x", "y"].transpose["x"]]  # pylint: disable=pointless-statement


def test_transpose_move_to_back():
  check_expr(
      expr=ts.d["y", "x"].transpose[-1],
      expected_repr="d['y','x'].transpose[-1]",
      before=ts.IndexTransform(input_labels=["x", "y", "z"]),
      after=ts.IndexTransform(
          input_labels=["z", "y", "x"],
          output=[
              ts.OutputIndexMap(input_dimension=2),
              ts.OutputIndexMap(input_dimension=1),
              ts.OutputIndexMap(input_dimension=0),
          ],
      ),
  )


def test_transpose_equality():
  check_expr_equality([
      ts.d[1, 2].transpose[:],
      ts.d[1, 2].transpose[2:3],
      ts.d[1, 2].transpose[-1],
  ])


def test_mark_bounds_implicit_true():
  check_expr(
      expr=ts.d[0, 2].mark_bounds_implicit[True],
      before=ts.IndexTransform(implicit_lower_bounds=[True, False, False],
                               implicit_upper_bounds=[False, False, True]),
      after=ts.IndexTransform(implicit_lower_bounds=[True, False, True],
                              implicit_upper_bounds=[True, False, True]),
  )


def test_mark_bounds_implicit_false():
  check_expr(
      expr=ts.d[0, 2].mark_bounds_implicit[False],
      before=ts.IndexTransform(implicit_lower_bounds=[True, True, False],
                               implicit_upper_bounds=[False, True, True]),
      after=ts.IndexTransform(implicit_lower_bounds=[False, True, False],
                              implicit_upper_bounds=[False, True, False]),
  )


def test_mark_bounds_implicit_none_true():
  check_expr(
      expr=ts.d[0, 2].mark_bounds_implicit[:True],
      before=ts.IndexTransform(implicit_lower_bounds=[True, False, False],
                               implicit_upper_bounds=[False, False, True]),
      after=ts.IndexTransform(implicit_lower_bounds=[True, False, False],
                              implicit_upper_bounds=[True, False, True]),
  )


def test_mark_bounds_implicit_none_false():
  check_expr(
      expr=ts.d[0, 2].mark_bounds_implicit[:False],
      before=ts.IndexTransform(implicit_lower_bounds=[True, True, False],
                               implicit_upper_bounds=[False, True, True]),
      after=ts.IndexTransform(implicit_lower_bounds=[True, True, False],
                              implicit_upper_bounds=[False, True, False]),
  )


def test_mark_bounds_implicit_true_none():
  check_expr(
      expr=ts.d[0, 2].mark_bounds_implicit[True:],
      before=ts.IndexTransform(implicit_lower_bounds=[True, False, False],
                               implicit_upper_bounds=[False, False, True]),
      after=ts.IndexTransform(implicit_lower_bounds=[True, False, True],
                              implicit_upper_bounds=[False, False, True]),
  )


def test_mark_bounds_implicit_false_none():
  check_expr(
      expr=ts.d[0, 2].mark_bounds_implicit[False:],
      before=ts.IndexTransform(implicit_lower_bounds=[True, True, False],
                               implicit_upper_bounds=[False, True, True]),
      after=ts.IndexTransform(implicit_lower_bounds=[False, True, False],
                              implicit_upper_bounds=[False, True, True]),
  )


def test_mark_bounds_implicit_false_true():
  check_expr(
      expr=ts.d[0, 2].mark_bounds_implicit[False:True],
      before=ts.IndexTransform(implicit_lower_bounds=[True, True, False],
                               implicit_upper_bounds=[False, False, True]),
      after=ts.IndexTransform(implicit_lower_bounds=[False, True, False],
                              implicit_upper_bounds=[True, False, True]),
  )


def test_mark_bounds_implicit_equality():
  check_expr_equality([
      ts.d[1, 2].mark_bounds_implicit[:],
      ts.d[1, 2].mark_bounds_implicit[True:],
      ts.d[1, 2].mark_bounds_implicit[False:],
      ts.d[1, 2].mark_bounds_implicit[:True],
      ts.d[1, 2].mark_bounds_implicit[:False],
      ts.d[1, 2].mark_bounds_implicit[True],
      ts.d[1, 2].mark_bounds_implicit[False],
      ts.d[1, 2].mark_bounds_implicit[True:False],
      ts.d[1, 2].mark_bounds_implicit[False:True],
  ])


def test_index_integer():
  check_expr(
      expr=ts.d["x", "y"][2, 3],
      expected_repr="d['x','y'][2,3]",
      before=ts.IndexTransform(input_shape=[15, 20], input_labels=["x", "y"]),
      after=ts.IndexTransform(
          input_rank=0,
          output=[ts.OutputIndexMap(offset=2),
                  ts.OutputIndexMap(offset=3)],
      ),
  )


def test_index_integer_non_scalar():
  x = ts.IndexTransform(input_shape=[15, 20], input_labels=["x", "y"])
  check_expr(
      expr=ts.d["x"][2,],
      expected_repr="d['x'][2,]",
      before=x,
      after=ts.IndexTransform(
          domain=x.domain[("y",)],
          output=[
              ts.OutputIndexMap(offset=2),
              ts.OutputIndexMap(input_dimension=0)
          ],
      ),
  )


def test_index_slice():
  check_expr(
      expr=ts.d["x", "y"][(1, 2):(7, 8)],
      expected_repr="d['x','y'][1:7,2:8]",
      before=ts.IndexTransform(input_shape=[15, 20], input_labels=["x", "y"]),
      after=ts.IndexTransform(
          input_inclusive_min=[1, 2],
          input_exclusive_max=[7, 8],
          input_labels=["x", "y"],
      ),
  )


def test_index_slice_incompatible_stop():
  with pytest.raises(
      IndexError,
      match=re.escape(
          "stop=[7,8,9] (rank 3) is incompatible with start=[1,2] (rank 2)"),
  ):
    ts.d[:][(1, 2):(7, 8, 9)]  # pylint: disable=pointless-statement


def test_index_slice_incompatible_step():
  with pytest.raises(
      IndexError,
      match=re.escape(
          "step=[9,10,11] (rank 3) is incompatible with stop=[7,8] (rank 2)"),
  ):
    ts.d[:][(1, 2):(7, 8):(9, 10, 11)]  # pylint: disable=pointless-statement


def test_index_slice_invalid_start():
  with pytest.raises(
      TypeError,
      match=re.escape(
          "slice indices must be integers or None or have an __index__ method"),
  ):
    ts.d[:]["a":3]  # pylint: disable=pointless-statement


def test_index_slice_invalid_stop():
  with pytest.raises(TypeError):
    ts.d[:][3:"a"]  # pylint: disable=pointless-statement


def test_index_slice_invalid_step():
  with pytest.raises(TypeError):
    ts.d[:][3:5:"a"]  # pylint: disable=pointless-statement


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
  check_expr(
      expr=ts.d["x", "y"][(1, 2):(8, 9):2],
      expected_repr="d['x','y'][1:8:2,2:9:2]",
      before=ts.IndexTransform(input_shape=[15, 20], input_labels=["x", "y"]),
      after=ts.IndexTransform(
          input_inclusive_min=[0, 1],
          input_inclusive_max=[3, 4],
          input_labels=["x", "y"],
          output=[
              ts.OutputIndexMap(input_dimension=0, offset=1, stride=2),
              ts.OutputIndexMap(input_dimension=1, stride=2),
          ],
      ),
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
      IndexError, match=re.escape(
          "Indexing expression requires 3 dimensions, and cannot be applied to "
          "a domain of rank 2")):
    x[1, 2, 3]  # pylint: disable=pointless-statement


def test_dimension_selection_index_too_many_ops():
  x = ts.IndexTransform(input_rank=2)
  with pytest.raises(
      IndexError,
      match=re.escape(
          "Indexing expression requires 3 dimensions but selection has 2 "
          "dimensions"),
  ):
    x[ts.d[0, 1][1, 2, 3]]  # pylint: disable=pointless-statement


def test_dimension_selection_chained_newaxis():
  x = ts.IndexTransform(input_rank=2)
  with pytest.raises(
      IndexError,
      match=re.escape(
          "tensorstore.newaxis (`None`) not valid in chained indexing "
          "operations"),
  ):
    x[ts.d[:].label["x", "y"][ts.newaxis, ...]]  # pylint: disable=pointless-statement


def test_dimension_selection_index_arrays_non_consecutive():
  check_expr(
      before=ts.IndexTransform(input_labels=["a", "b", "c", "d"]),
      expr=ts.d["a", "c", "d"][[[1, 2, 3], [4, 5, 6]], :, [6, 7, 8]],
      after=ts.IndexTransform(
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
      ),
  )


def test_dimension_selection_index_arrays_consecutive():
  check_expr(
      before=ts.IndexTransform(input_labels=["a", "b", "c", "d"]),
      expr=ts.d["a", "c", "d"][..., [[1, 2, 3], [4, 5, 6]], [6, 7, 8]],
      after=ts.IndexTransform(
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
      ),
  )


def test_dimension_selection_bool_arrays_non_consecutive():
  check_expr(
      before=ts.IndexTransform(input_labels=["a", "b", "c", "d"]),
      expr=ts.d["a", "b", "c",
                "d"][[[False, True, True], [False, False, False]], :,
                     [False, False, True, True]],
      after=ts.IndexTransform(
          domain=[ts.Dim(size=2), ts.Dim(label="c")],
          output=[
              ts.OutputIndexMap(index_array=[[0], [0]]),
              ts.OutputIndexMap(index_array=[[1], [2]]),
              ts.OutputIndexMap(input_dimension=1),
              ts.OutputIndexMap(index_array=[[2], [3]]),
          ],
      ),
  )


def test_dimension_selection_bool_arrays_consecutive():
  check_expr(
      before=ts.IndexTransform(input_labels=["a", "b", "c", "d"]),
      expr=ts.d["a", "b", "c",
                "d"][:, [[False, True, True], [False, False, False]],
                     [False, False, True, True]],
      after=ts.IndexTransform(
          domain=[ts.Dim(label="a"), ts.Dim(size=2)],
          output=[
              ts.OutputIndexMap(input_dimension=0),
              ts.OutputIndexMap(index_array=[[0, 0]]),
              ts.OutputIndexMap(index_array=[[1, 2]]),
              ts.OutputIndexMap(index_array=[[2, 3]]),
          ],
      ),
  )


def test_dimension_selection_oindex_bool_arrays():
  check_expr(
      before=ts.IndexTransform(input_labels=["a", "b", "c", "d"]),
      expr=ts.d["a", "b", "c",
                "d"].oindex[[[False, True, True], [False, False, False]], :,
                            [True, False, True, True]],
      after=ts.IndexTransform(
          domain=[ts.Dim(size=2),
                  ts.Dim(label="c"),
                  ts.Dim(size=3)],
          output=[
              ts.OutputIndexMap(index_array=[[[0]], [[0]]]),
              ts.OutputIndexMap(index_array=[[[1]], [[2]]]),
              ts.OutputIndexMap(input_dimension=1),
              ts.OutputIndexMap(index_array=[[[0, 2, 3]]]),
          ],
      ),
  )


def test_dimension_selection_vindex_bool_arrays_non_consecutive():
  check_expr(
      before=ts.IndexTransform(input_labels=["a", "b", "c", "d"]),
      expr=ts.d["a", "b", "c",
                "d"].vindex[[[False, True, True], [False, False, False]], :,
                            [False, False, True, True]],
      after=ts.IndexTransform(
          domain=[ts.Dim(size=2), ts.Dim(label="c")],
          output=[
              ts.OutputIndexMap(index_array=[[0], [0]]),
              ts.OutputIndexMap(index_array=[[1], [2]]),
              ts.OutputIndexMap(input_dimension=1),
              ts.OutputIndexMap(index_array=[[2], [3]]),
          ],
      ),
  )


def test_dimension_selection_vindex_repr():
  check_expr(
      expr=ts.d[:].vindex[1, 2],
      expected_repr="d[:].vindex[1,2]",
  )


def test_dimension_selection_oindex_repr():
  check_expr(
      expr=ts.d[:].oindex[1, 2],
      expected_repr="d[:].oindex[1,2]",
  )


def test_dimension_selection_index_repr():
  check_expr(
      expr=ts.d[:][1, 2],
      expected_repr="d[:][1,2]",
  )


def test_dimension_selection_vindex_bool_arrays_consecutive():
  check_expr(
      before=ts.IndexTransform(input_labels=["a", "b", "c", "d"]),
      expr=ts.d["a", "b", "c",
                "d"].vindex[:, [[False, True, True], [False, False, False]],
                            [False, False, True, True]],
      after=ts.IndexTransform(
          domain=[ts.Dim(size=2), ts.Dim(label="a")],
          output=[
              ts.OutputIndexMap(input_dimension=1),
              ts.OutputIndexMap(index_array=[[0], [0]]),
              ts.OutputIndexMap(index_array=[[1], [2]]),
              ts.OutputIndexMap(index_array=[[2], [3]]),
          ],
      ),
  )


def test_dimension_selection_oindex_index_arrays():
  check_expr(
      before=ts.IndexTransform(input_labels=["a", "b", "c", "d"]),
      expr=ts.d["a", "c", "d"].oindex[[[1, 2, 3], [4, 5, 6]], :, [6, 7, 8]],
      after=ts.IndexTransform(
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
      ),
  )


def test_dimension_selection_vindex_index_arrays_non_consecutive():
  check_expr(
      before=ts.IndexTransform(input_labels=["a", "b", "c", "d"]),
      expr=ts.d["a", "c", "d"].vindex[[[1, 2, 3], [4, 5, 6]], :, [6, 7, 8]],
      after=ts.IndexTransform(
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
      ),
  )


def test_dimension_selection_vindex_index_arrays_consecutive():
  check_expr(
      before=ts.IndexTransform(input_labels=["a", "b", "c", "d"]),
      expr=ts.d["a", "c", "d"].vindex[..., [[1, 2, 3], [4, 5, 6]], [6, 7, 8]],
      after=ts.IndexTransform(
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
      ),
  )


def test_dimension_selection_dimrange_index():
  check_expr(
      before=ts.IndexTransform(input_labels=["a", "b", "c"]),
      expr=ts.d["c", :2][:5, 1, 2],
      after=ts.IndexTransform(
          input_labels=["c"],
          input_exclusive_max=[5],
          output=[
              ts.OutputIndexMap(1),
              ts.OutputIndexMap(2),
              ts.OutputIndexMap(input_dimension=0),
          ],
      ),
  )


def test_dimension_selection_dimrange_index_invalid_start():
  x = ts.IndexTransform(input_rank=3)
  expr = ts.d[5:][...]
  with pytest.raises(
      IndexError,
      match=re.escape("Dimension index 5 is outside valid range [-3, 3)"),
  ):
    x[expr]  # pylint: disable=pointless-statement


def test_dimension_selection_dimrange_index_invalid_stop():
  x = ts.IndexTransform(input_rank=3)
  expr = ts.d[:5][...]
  with pytest.raises(
      IndexError,
      match=re.escape(
          "Dimension exclusive stop index 5 is outside valid range [-4, 3]"),
  ):
    x[expr]  # pylint: disable=pointless-statement


def test_dimension_selection_duplicate_index():
  x = ts.IndexTransform(input_rank=3)
  expr = ts.d[1, 1][...]
  with pytest.raises(IndexError,
                     match=re.escape("Dimension 1 specified more than once")):
    x[expr]  # pylint: disable=pointless-statement


def test_dimension_selection_duplicate_index_label():
  x = ts.IndexTransform(input_labels=["x", "y", "z"])
  expr = ts.d[1, "y"][...]
  with pytest.raises(IndexError,
                     match=re.escape("Dimension 1 specified more than once")):
    x[expr]  # pylint: disable=pointless-statement


def test_dimension_selection_duplicate_index_label_newaxis():
  x = ts.IndexTransform(input_labels=["x", "y", "z"])
  expr = ts.d[0, 2, "y"][ts.newaxis, ...]
  with pytest.raises(IndexError,
                     match=re.escape("Dimension 2 specified more than once")):
    x[expr]  # pylint: disable=pointless-statement


def test_dimension_selection_index_label_newaxis():
  x = ts.IndexTransform(input_labels=["x", "y", "z"])
  expr = ts.d[0, "y"][ts.newaxis, ts.newaxis]
  with pytest.raises(
      IndexError,
      match=re.escape(
          "Dimensions specified by label cannot be used with newaxis"),
  ):
    x[expr]  # pylint: disable=pointless-statement


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
          "Zero-rank bool array incompatible with outer indexing of a "
          "dimension selection"),
  ):
    x[ts.d[:].oindex[..., True]]  # pylint: disable=pointless-statement


def test_numpy_indexing_equality():
  check_expr_equality([
      ts.d[2:][1],
      ts.d[2:][2],
      ts.d[2:][2, 3],
      ts.d[2:][2, 4],
      ts.d[2:][2:4],
      ts.d[2:][1:4],
      ts.d[2:][1:5],
      ts.d[2:][1:5:6],
      ts.d[2:][1:5:7],
      ts.d[2:][1:5:7, 2:8],
      ts.d[2:][1:5:7, 2:8, ...],
      ts.d[2:][1:5:7, 2:8, None],
      ts.d[2:][[1, 2, 3]],
      ts.d[2:].vindex[[1, 2, 3]],
      ts.d[2:].oindex[[1, 2, 3]],
  ])


def test_label_without_dimension_selection():
  before = ts.IndexTransform(input_rank=3)
  after = ts.IndexTransform(input_labels=["x", "y", "z"])
  assert before.label["x", "y", "z"] == after
  assert before.domain.label["x", "y", "z"] == after.domain


def test_translate_to_without_dimension_selection():
  before = ts.IndexTransform(input_shape=[20, 30])
  after = ts.IndexTransform(
      input_inclusive_min=[20, 20], input_shape=[20, 30], output=[
          ts.OutputIndexMap(input_dimension=0, offset=-20),
          ts.OutputIndexMap(input_dimension=1, offset=-20),
      ])
  assert before.translate_to[20] == after
  assert before.domain.translate_to[20] == after.domain


def test_translate_by_without_dimension_selection():
  before = ts.IndexTransform(input_shape=[20, 30])
  after = ts.IndexTransform(
      input_inclusive_min=[20, 20], input_shape=[20, 30], output=[
          ts.OutputIndexMap(input_dimension=0, offset=-20),
          ts.OutputIndexMap(input_dimension=1, offset=-20),
      ])
  assert before.translate_by[20] == after
  assert before.domain.translate_by[20] == after.domain


def test_translate_backward_by_without_dimension_selection():
  before = ts.IndexTransform(input_shape=[20, 30])
  after = ts.IndexTransform(
      input_inclusive_min=[20, 20], input_shape=[20, 30], output=[
          ts.OutputIndexMap(input_dimension=0, offset=-20),
          ts.OutputIndexMap(input_dimension=1, offset=-20),
      ])
  assert before.translate_backward_by[-20] == after
  assert before.domain.translate_backward_by[-20] == after.domain


def test_mark_bounds_implicit_without_dimension_selection():
  before = ts.IndexTransform(input_shape=[20, 30])
  after = ts.IndexTransform(
      input_shape=[20, 30],
      implicit_lower_bounds=[True, True],
      implicit_upper_bounds=[True, True],
  )
  assert before.mark_bounds_implicit[True] == after
  assert before.domain.mark_bounds_implicit[True] == after.domain


def test_transpose_without_dimension_selection():
  before = ts.IndexTransform(input_shape=[20, 30])
  after = ts.IndexTransform(
      input_shape=[30, 20],
      output=[
          ts.OutputIndexMap(input_dimension=1),
          ts.OutputIndexMap(input_dimension=0)
      ],
  )
  assert before.transpose([1, 0]) == after
  assert before.domain.transpose([1, 0]) == after.domain
  assert before.transpose() == after
  assert before.domain.transpose() == after.domain
