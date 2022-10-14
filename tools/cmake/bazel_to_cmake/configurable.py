# Copyright 2022 The TensorStore Authors
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
"""Defines `select`-related data structures."""

# pylint: disable=relative-beyond-top-level

import operator
from typing import TypeVar, Generic, List, Union, cast, Dict, Callable

from .label import Label

T = TypeVar("T")


class _ConfigurableBase(Generic[T]):
  """Base class for `Select` and `SelectExpression`.

  Provides operator definitions.
  """

  def __add__(self, other: "Configurable[T]") -> "Configurable[T]":
    return SelectExpression(operator.add,
                            cast(List[Configurable[T]], [self, other]))

  def __radd__(self, other: "Configurable[T]") -> "Configurable[T]":
    return SelectExpression(operator.add,
                            cast(List[Configurable[T]], [other, self]))


class Select(_ConfigurableBase[T]):
  """Represents a parsed (but not evaluated) `select` expression."""

  def __init__(self, conditions: Dict[Label, T]):
    self.conditions = conditions


class SelectExpression(_ConfigurableBase[T]):
  """Represents an expression true involving `select` expressions."""

  def __init__(
      self,
      op: Callable[["Configurable[T]", "Configurable[T]"], T],
      operands: List["Configurable[T]"],
  ):
    self.op = op
    self.operands = operands


Configurable = Union[T, Select[T], SelectExpression[T]]
