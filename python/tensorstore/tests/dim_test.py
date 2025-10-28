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
"""Tests for tensorstore.Dim"""

import pickle
import threading
import time

import pytest
import tensorstore as ts


def test_unbounded() -> None:
  x = ts.Dim()
  assert x.inclusive_min == -ts.inf
  assert x.inclusive_max == +ts.inf
  assert not x.finite


def test_closed() -> None:
  x = ts.Dim(inclusive_min=3, inclusive_max=5)
  assert x.inclusive_min == 3
  assert x.inclusive_max == 5
  assert x.exclusive_min == 2
  assert x.exclusive_max == 6
  assert 3 in x
  assert 5 in x
  assert 2 not in x
  assert 6 not in x
  assert ts.Dim(inclusive_min=3, inclusive_max=4) in x
  assert ts.Dim(inclusive_min=3, inclusive_max=6) not in x
  y = ts.Dim(inclusive_min=3, inclusive_max=5)
  z = ts.Dim(inclusive_min=3, inclusive_max=6)
  assert x == y
  assert x != z
  assert x.size == 3
  assert len(x) == 3
  assert x.empty is False
  assert repr(x) == "Dim(inclusive_min=3, exclusive_max=6)"
  assert str(x) == "[3, 6)"
  assert list(x) == [3, 4, 5]

  x1 = ts.Dim(inclusive_max=5)
  assert x1.inclusive_min == -ts.inf
  assert x1.inclusive_max == 5
  assert x1 == ts.Dim(inclusive_min=None, inclusive_max=5)

  with pytest.raises(ValueError):
    ts.Dim(inclusive_min=3, inclusive_max=1)


def test_half_open() -> None:
  x = ts.Dim(3, 5)
  assert x.inclusive_min == 3
  assert x.exclusive_max == 5
  y = ts.Dim(inclusive_min=3, exclusive_max=5)
  assert x == y
  z = ts.Dim(exclusive_max=5)
  assert z.inclusive_min == -ts.inf
  assert z.exclusive_max == 5
  assert z == ts.Dim(inclusive_min=None, exclusive_max=5)

  with pytest.raises(ValueError):
    ts.Dim(inclusive_min=3, exclusive_max=1)


def test_sized() -> None:
  x = ts.Dim(inclusive_min=3, size=10)
  assert x.inclusive_min == 3
  assert x.size == 10
  x1 = ts.Dim(size=10)
  assert x1.inclusive_min == 0
  assert x1.size == 10
  assert x1 == ts.Dim(inclusive_min=None, size=10)

  y = ts.Dim(size=None)
  assert y.inclusive_min == 0
  assert y.inclusive_max == +ts.inf

  with pytest.raises(ValueError):
    ts.Dim(inclusive_min=3, size=-3)


def test_intersect() -> None:
  a = ts.Dim(inclusive_min=1, exclusive_max=5, label="x")
  b = ts.Dim(size=3)
  x = a.intersect(b)
  assert x.inclusive_min == 1
  assert x.exclusive_max == 3
  assert x.label == "x"

  with pytest.raises(ValueError):
    a.intersect(ts.Dim(size=3, label="y"))


def test_hull() -> None:
  a = ts.Dim(inclusive_min=1, exclusive_max=5, label="x")
  b = ts.Dim(size=3)
  x = a.hull(b)
  assert x.inclusive_min == 0
  assert x.exclusive_max == 5
  assert x.label == "x"

  with pytest.raises(ValueError):
    a.hull(ts.Dim(size=3, label="y"))


def test_pickle() -> None:
  x = ts.Dim(inclusive_min=3, size=10)
  assert pickle.loads(pickle.dumps(x)) == x


def test_dim_concurrent() -> None:
  """Tests concurrent access to Dim properties."""
  dim = ts.Dim()

  stop = threading.Event()

  def read_props() -> None:
    while not stop.is_set():
      _ = dim.inclusive_min
      _ = dim.implicit_lower
      _ = dim.implicit_upper
      _ = dim.label
      _ = dim == ts.Dim()
      _ = f"{dim}"
      _ = repr(dim)

  def update_props() -> None:
    time.sleep(0.01)
    i = 0
    while not stop.is_set():
      if (i % 2) == 0:
        dim.implicit_lower = True
        dim.implicit_upper = False
        dim.label = "x"
      else:
        dim.implicit_lower = False
        dim.implicit_upper = True
        dim.label = ""
      i += 1

  threads = []
  for _ in range(4):
    threads.append(threading.Thread(target=read_props))
    threads.append(threading.Thread(target=update_props))

  for t in threads:
    t.start()

  time.sleep(0.3)
  stop.set()

  for t in threads:
    t.join()
