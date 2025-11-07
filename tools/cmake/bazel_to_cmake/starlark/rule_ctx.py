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

import pathlib
from typing import NamedTuple

from ..util import write_file_if_not_already_equal
from .common_providers import ConditionProvider
from .invocation_context import InvocationContext
from .label import Label


class RuleFile(NamedTuple):
  """Represents a file for use by rule implementations."""

  path: str


class RuleCtxAttr:
  pass


class RuleCtxActions:
  """https://bazel.build/rules/lib/builtins/actions"""

  def __init__(self, ctx: "RuleCtx"):
    self._ctx = ctx

  def write(
      self,
      output: RuleFile,
      content: str,
      is_executable: bool = False,
      *,
      mnemonic: str | None = None,
  ):
    del is_executable
    del mnemonic
    write_file_if_not_already_equal(
        pathlib.PurePath(output.path), content.encode("utf-8")
    )


class RuleCtx:

  def __init__(self, context: InvocationContext, label: Label):
    self._context: InvocationContext = context
    self.label = label
    self.attr = RuleCtxAttr()
    self.outputs = RuleCtxAttr()
    self.actions = RuleCtxActions(self)

  def target_platform_has_constraint(self, constraint: ConditionProvider):
    return constraint.value
