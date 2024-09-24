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
"""CMake implementation of native Bazel build rules.

To see how bazel implements rules in Java, see:
https://github.com/bazelbuild/bazel/tree/master/src/main/java/com/google/devtools/build/lib/packages

And to see the native skylark implementations, see:
https://github.com/bazelbuild/bazel/tree/master/src/main/starlark/builtins_bzl/common
"""

# pylint: disable=relative-beyond-top-level,invalid-name,missing-function-docstring,g-long-lambda

from typing import Dict, List, Optional

from .evaluation import EvaluationState
from .provider_util import ProviderCollection
from .starlark import rule  # pylint: disable=unused-import
from .starlark.bazel_build_file import register_native_build_rule
from .starlark.bazel_target import TargetId
from .starlark.common_providers import BuildSettingProvider
from .starlark.common_providers import ConditionProvider
from .starlark.common_providers import ConstraintSettingInfo
from .starlark.common_providers import ConstraintValueInfo
from .starlark.common_providers import PlatformInfo
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.provider import TargetInfo


@register_native_build_rule
def constraint_setting(
    self: InvocationContext,
    name: str,
    default_constraint_value: Optional[str] = None,
    **kwargs,
):
  del kwargs
  context = self.snapshot()
  label = context.resolve_target(name)
  if default_constraint_value is not None:
    default_constraint_value = context.resolve_target(default_constraint_value)
  # Always analyze.
  context.add_analyzed_target(
      context.resolve_target(name),
      TargetInfo(ConstraintSettingInfo(label, default_constraint_value)),
  )


@register_native_build_rule
def constraint_value(
    self: InvocationContext,
    name: str,
    constraint_setting: Optional[RelativeLabel],
    **kwargs,
):
  del kwargs
  context = self.snapshot()
  _label = context.resolve_target(name)
  _constraint_setting = context.resolve_target_or_label(constraint_setting)
  context.add_analyzed_target(
      _label,
      TargetInfo(ConstraintValueInfo(_constraint_setting, _label)),
  )


# https://bazel.build/reference/be/platforms-and-toolchains#platform
@register_native_build_rule
def platform(
    self: InvocationContext,
    name: str,
    constraint_values: Optional[List[RelativeLabel]] = None,
    exec_properties: Optional[Dict[str, str]] = None,
    flags: Optional[List[str]] = None,
    parents: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  del kwargs
  if flags:
    print(f"WARNING: platform flags not supported: {flags}")
  if exec_properties:
    print(f"WARNING: exec_properties not supported: {exec_properties}")

  context = self.snapshot()
  target = context.resolve_target(name)
  _constraint_values = context.resolve_target_or_label_list(constraint_values)
  _parents = context.resolve_target_or_label_list(parents)

  context.add_rule(
      target,
      lambda: _platform_impl(
          context,
          target,
          _constraint_values,
          _parents,
      ),
      analyze_by_default=True,
  )


def _platform_impl(
    _context: InvocationContext,
    _target: TargetId,
    constraint_values: Optional[List[TargetId]] = None,
    parents: Optional[List[TargetId]] = None,
):
  # Bazel to CMake doesn't have a way to support platform flags.
  state = _context.access(EvaluationState)

  collector = ProviderCollection()
  for x in parents:
    collector.collect(x, state.get_target_info(x))

  constraints = set()
  for x in collector.items(PlatformInfo):
    constraints.update(x.constraints)
  constraints.update(constraint_values)

  workspace_values = _context.access(EvaluationState).workspace.values

  def _evaluate() -> bool:
    nonlocal constraints
    for c in constraints:
      assert isinstance(c, TargetId)
      c_info = _context.get_target_info(c)
      if c_info.get(ConditionProvider) is not None:
        if not c_info[ConditionProvider].value:
          return False
      elif c_info.get(BuildSettingProvider) is not None:
        if not c_info[BuildSettingProvider].value:
          return False
      elif c_info.get(ConstraintValueInfo) is not None:
        cv = c_info[ConstraintValueInfo]
        if (
            cv.constraint.as_label(),
            cv.label.as_label(),
        ) not in workspace_values:
          return False
      else:
        return False
    return True

  condition = _evaluate()
  _context.add_analyzed_target(
      _target,
      TargetInfo(
          PlatformInfo(_target, list(sorted(constraints))),
          ConditionProvider(condition),
      ),
  )
