# Copyright 2026 The TensorStore Authors
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
"""OrderedSet implementation."""

# pylint: disable=g-importing-member,missing-class-docstring

from __future__ import annotations

from collections.abc import Iterable, Iterator, MutableSet
from typing import Generic, TypeVar

_T = TypeVar("_T")


class OrderedSet(MutableSet[_T], Generic[_T]):
  __slots__ = ("_dict",)

  def __init__(self, iterable: Iterable[_T] | None = None):
    self._dict = {}
    if iterable is not None:
      self.update(iterable)

  def __str__(self) -> str:
    return "OrderedSet(%s)" % ", ".join(repr(x) for x in self)

  def __repr__(self) -> str:
    return self.__str__()

  def update(self, iterable: Iterable[_T]):
    if isinstance(iterable, OrderedSet):
      self._dict.update(iterable._dict)
    else:
      self._dict.update((value, None) for value in iterable)

  def __contains__(self, value: object) -> bool:
    return value in self._dict

  def __iter__(self) -> Iterator[_T]:
    return iter(self._dict)

  def __len__(self) -> int:
    return len(self._dict)

  def add(self, value: _T):
    """Add an element."""
    self._dict[value] = None

  def discard(self, value: _T):
    """Remove an element. Do not raise an exception if absent."""
    if value in self._dict:
      del self._dict[value]

  def clear(self):
    """Remove all elements from this set."""
    self._dict.clear()

  def remove(self, value: _T):
    """Remove an element. If not a member, raise a KeyError."""
    del self._dict[value]

  def intersection(self, other: Iterable[_T]) -> OrderedSet[_T]:
    other = set(other)
    res = OrderedSet(filter(other.__contains__, self))
    return res

  def union(self, other: Iterable[_T]) -> OrderedSet[_T]:
    res = OrderedSet(self)
    res.update(other)
    return res

  def copy(self) -> OrderedSet[_T]:
    return OrderedSet(self)
