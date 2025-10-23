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
"""Tests of tensorstore.IndexDomain"""

import pickle

import pytest
import tensorstore as ts


def test_init_rank() -> None:
  x = ts.IndexDomain(rank=2)
  assert x.rank == 2
  assert x.ndim == 2
  assert x.inclusive_min == (-ts.inf,) * 2
  assert x.inclusive_max == (+ts.inf,) * 2
  assert x.exclusive_max == (+ts.inf + 1,) * 2
  assert x.shape == (2 * ts.inf + 1,) * 2
  assert x.labels == ("", "")
  assert x.implicit_lower_bounds == (True, True)
  assert x.implicit_upper_bounds == (True, True)

  with pytest.raises(ValueError):
    ts.IndexDomain(rank=33)


def test_init_inclusive_min() -> None:
  x = ts.IndexDomain(inclusive_min=[1, 2])
  assert x.rank == 2
  assert x.inclusive_min == (1, 2)
  assert x.inclusive_max == (+ts.inf,) * 2
  assert x.exclusive_max == (+ts.inf + 1,) * 2
  assert x.labels == ("", "")
  assert x.implicit_lower_bounds == (False, False)
  assert x.implicit_upper_bounds == (True, True)

  with pytest.raises(ValueError):
    ts.IndexDomain(inclusive_min=[1] * 33)


def test_init_exclusive_max() -> None:
  x = ts.IndexDomain(exclusive_max=[1, 2])
  assert x.rank == 2
  assert x.inclusive_min == (-ts.inf,) * 2
  assert x.exclusive_max == (1, 2)
  assert x.labels == ("", "")
  assert x.implicit_lower_bounds == (True, True)
  assert x.implicit_upper_bounds == (False, False)


def test_init_inclusive_max() -> None:
  x = ts.IndexDomain(inclusive_max=[1, 2])
  assert x.rank == 2
  assert x.inclusive_min == (-ts.inf,) * 2
  assert x.inclusive_max == (1, 2)
  assert x.labels == ("", "")
  assert x.implicit_lower_bounds == (True, True)
  assert x.implicit_upper_bounds == (False, False)


def test_init_shape() -> None:
  x = ts.IndexDomain(shape=[1, 2])
  assert x.rank == 2
  assert x.inclusive_min == (0,) * 2
  assert x.exclusive_max == (1, 2)
  assert x.shape == (1, 2)
  assert x.labels == ("", "")
  assert x.implicit_lower_bounds == (False, False)
  assert x.implicit_upper_bounds == (False, False)


def test_init_labels() -> None:
  x = ts.IndexDomain(labels=["x", "y"])
  assert x.rank == 2
  assert x.inclusive_min == (-ts.inf,) * 2
  assert x.inclusive_max == (+ts.inf,) * 2
  assert x.labels == ("x", "y")
  assert x.implicit_lower_bounds == (True, True)
  assert x.implicit_upper_bounds == (True, True)


def test_init_implicit_lower_bounds() -> None:
  x = ts.IndexDomain(implicit_lower_bounds=[False, True])
  assert x.rank == 2
  assert x.inclusive_min == (-ts.inf,) * 2
  assert x.inclusive_max == (+ts.inf,) * 2
  assert x.exclusive_max == (+ts.inf + 1,) * 2
  assert x.labels == ("", "")
  assert x.implicit_lower_bounds == (False, True)
  assert x.implicit_upper_bounds == (True, True)


def test_init_rank_mismatch() -> None:
  with pytest.raises(
      ValueError,
      match=r"Rank specified by `inclusive_min` \(1\) does not "
      r"match rank specified by `rank` \(2\)",
  ):
    ts.IndexDomain(rank=2, inclusive_min=[1])


def test_init_duplicate_upper_bound() -> None:
  with pytest.raises(
      ValueError,
      match="Cannot specify both `exclusive_max` and `inclusive_max`",
  ):
    ts.IndexDomain(inclusive_max=[1, 2], exclusive_max=[1, 2])


def test_init_missing_rank() -> None:
  with pytest.raises(ValueError, match="Must specify `rank`"):
    ts.IndexDomain()


def test_init_implicit_upper_bounds() -> None:
  x = ts.IndexDomain(implicit_upper_bounds=[False, True])
  assert x.rank == 2
  assert x.inclusive_min == (-ts.inf,) * 2
  assert x.inclusive_max == (+ts.inf,) * 2
  assert x.exclusive_max == (+ts.inf + 1,) * 2
  assert x.labels == ("", "")
  assert x.implicit_lower_bounds == (True, True)
  assert x.implicit_upper_bounds == (False, True)


def test_intersect() -> None:
  a = ts.IndexDomain(
      inclusive_min=[0, 1, 2, 3],
      exclusive_max=[2, 4, 5, 6],
      labels=["x", "y", "", ""],
      implicit_lower_bounds=[False, False, False, True],
      implicit_upper_bounds=[False, False, False, True],
  )
  b = ts.IndexDomain(
      inclusive_min=[0, 0, 0, 0],
      exclusive_max=[2, 2, 3, 4],
      implicit_upper_bounds=[True, False, True, True],
  )
  x = a.intersect(b)
  assert x.inclusive_min == (0, 1, 2, 3)
  assert x.exclusive_max == (2, 2, 3, 4)
  assert x.labels == ("x", "y", "", "")
  assert x.implicit_upper_bounds == (False, False, True, True)
  assert x.implicit_lower_bounds == (False, False, False, True)


def test_hull() -> None:
  a = ts.IndexDomain(
      inclusive_min=[0, 1, 2, 3],
      exclusive_max=[2, 4, 5, 6],
      labels=["x", "y", "", ""],
      implicit_lower_bounds=[False, False, False, True],
      implicit_upper_bounds=[False, False, False, True],
  )
  b = ts.IndexDomain(
      inclusive_min=[0, 0, 0, 0],
      exclusive_max=[2, 2, 3, 4],
      implicit_upper_bounds=[True, False, True, True],
  )
  x = a.hull(b)
  assert x.inclusive_min == (0, 0, 0, 0)
  assert x.exclusive_max == (2, 4, 5, 6)
  assert x.labels == ("x", "y", "", "")
  assert x.implicit_upper_bounds == (False, False, False, True)
  assert x.implicit_lower_bounds == (False, False, False, False)


def test_getitem_index() -> None:
  d = ts.IndexDomain(labels=["x", "y"], shape=[3, 4])
  assert d["x"] == ts.Dim(label="x", size=3)
  assert d["y"] == ts.Dim(label="y", size=4)
  assert d[1] == ts.Dim(label="y", size=4)
  assert d["x", "y"] == d
  assert d["y", "x"] == ts.IndexDomain([d[1], d[0]])
  assert d[::-1] == ts.IndexDomain([d[1], d[0]])
  with pytest.raises(IndexError):
    d["z"]
  with pytest.raises(IndexError):
    d[2]
  with pytest.raises(ValueError):
    d[1:3]


def test_getitem_index_domain() -> None:
  d = ts.IndexDomain(labels=["x", "y", "z"], shape=[10, 11, 12])
  assert d[
      ts.IndexDomain(
          labels=["y", "x"], inclusive_min=[1, 2], exclusive_max=[6, 7]
      )
  ] == ts.IndexDomain(
      labels=["x", "y", "z"], inclusive_min=[2, 1, 0], exclusive_max=[7, 6, 12]
  )


def test_eq() -> None:
  a = ts.IndexDomain(labels=["x", "y"])
  b = ts.IndexDomain(labels=["x", "y"])
  c = ts.IndexDomain(labels=["x", "z"])
  assert a == b
  assert a != c


def test_transpose() -> None:
  a = ts.IndexDomain(labels=["x", "y"])
  b = ts.IndexDomain(labels=["y", "x"])

  assert a.T == b
  assert a.transpose() == b
  assert a.transpose([1, 0]) == b
  assert a.transpose(["y", "x"]) == b
  assert a.transpose(["y", 0]) == b


def test_transpose_errors() -> None:
  a = ts.IndexDomain(labels=["x", "y", "z"])
  with pytest.raises(ValueError):
    a.transpose([1, 0])
  with pytest.raises(ValueError):
    a.transpose(["x", "y", "y"])
  with pytest.raises(ValueError):
    a.transpose(["x", "y", "z", 1])
  with pytest.raises(ValueError):
    a.transpose(["x", "y", 1])


def test_pickle() -> None:
  x = ts.IndexDomain(
      labels=["x", "y"],
      inclusive_min=[1, 2],
      exclusive_max=[3, 4],
      implicit_lower_bounds=[True, False],
      implicit_upper_bounds=[False, True],
  )
  assert pickle.loads(pickle.dumps(x)) == x
