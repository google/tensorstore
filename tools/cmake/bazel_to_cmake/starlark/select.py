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

# pylint: disable=missing-function-docstring,relative-beyond-top-level

import operator
from typing import TypeVar, Generic, List, Union, cast, Dict, Callable

from .bazel_target import TargetId

T = TypeVar("T")
TestCondition = Callable[[TargetId], bool]


def _is_conditions_default(target: TargetId) -> bool:
  return target.package_name == "conditions" and target.target_name == "default"


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

  def evaluate(self, test_condition: TestCondition) -> T:
    raise ValueError("Bad Configurable")


class Select(_ConfigurableBase[T]):
  """Represents a parsed (but not evaluated) `select` expression."""

  def __init__(self, conditions: Dict[TargetId, T]):
    self.conditions = conditions

  def __repr__(self):
    return f"select({repr(self.conditions)})"

  def evaluate(self, test_condition: TestCondition) -> T:
    has_default = False
    default_value = None
    matches = []
    for condition, value in self.conditions.items():
      if _is_conditions_default(condition):
        has_default = True
        default_value = value
        continue
      if test_condition(condition):
        matches.append((condition, value))
    if len(matches) > 1:
      raise ValueError(f"More than one matching condition: {matches!r}")
    if len(matches) == 1:
      return matches[0][1]
    if has_default:
      return cast(T, default_value)
    raise ValueError("No matching condition")


class SelectExpression(_ConfigurableBase[T]):
  """Represents an expression true involving `select` expressions."""

  def __init__(
      self,
      op: Callable[["Configurable[T]", "Configurable[T]"], T],
      operands: List["Configurable[T]"],
  ):
    self.op = op
    self.operands = operands

  def __repr__(self):
    return f"SelectExpression({repr(self.op), repr(self.operands)})"

  def evaluate(self, test_condition: TestCondition) -> T:

    def _try_evaluate(t: Union[T, Configurable[T]]) -> T:
      if isinstance(t, _ConfigurableBase):
        return t.evaluate(test_condition)
      else:
        return t

    return self.op(*(_try_evaluate(operand) for operand in self.operands))


Configurable = Union[T, Select[T], SelectExpression[T]]
