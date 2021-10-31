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
"""Defines pytest fixtures for tensorstore tests."""

import gc
from typing import Any, List
import weakref

import pytest


@pytest.fixture
def gc_tester():
  """Tests that an object is garbage collected.

  Yields a function that should be called with objects that may be part of a
  reference cycle.

  After the test function returns, verifies that any objects specified to the
  yielded function are garbage collected.
  """

  weak_refs: List[weakref.ref] = []

  def add_ref(obj: Any) -> None:
    # PyPy does not support `gc.is_tracked`.
    if hasattr(gc, "is_tracked"):
      assert gc.is_tracked(obj)
    weak_refs.append(weakref.ref(obj))

  yield add_ref

  gc.collect()
  if hasattr(gc, "collect_step"):
    # PyPy may require additional encouragement to actually collect the garbage.
    for _ in range(100):
      gc.collect_step()
  for ref in weak_refs:
    assert ref() is None
