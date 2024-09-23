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
from .starlark import rule  # pylint: disable=unused-import
from .starlark.bazel_build_file import register_native_build_rule
from .starlark.bazel_target import TargetId
from .starlark.common_providers import BuildSettingProvider
from .starlark.common_providers import ConditionProvider
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.provider import TargetInfo


@register_native_build_rule
def config_setting(
    self: InvocationContext,
    name: str,
    constraint_values: Optional[List[RelativeLabel]] = None,
    flag_values: Optional[Dict[RelativeLabel, str]] = None,
    values: Optional[Dict[str, str]] = None,
    define_values: Optional[Dict[str, str]] = None,
    visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  del kwargs
  # Bazel ignores visibility for `config_setting` by default.  See
  # `--incompatible_enforce_config_setting_visibility` and
  # `--incompatible_config_setting_private_default_visibility`.
  del visibility
  context = self.snapshot()
  target = context.resolve_target(name)
  context.add_rule(
      target,
      lambda: _config_setting_impl(
          context,
          target,
          constraint_values=constraint_values,
          flag_values=flag_values,
          values=values,
          define_values=define_values,
      ),
      analyze_by_default=True,
  )


def _config_setting_impl(
    _context: InvocationContext,
    _target: TargetId,
    constraint_values: Optional[List[RelativeLabel]],
    flag_values: Optional[Dict[RelativeLabel, str]],
    values: Optional[Dict[str, str]],
    define_values: Optional[Dict[str, str]],
):
  def evaluate() -> bool:
    if flag_values:
      for flag, value in flag_values.items():
        flag_target = _context.resolve_target_or_label(flag)
        flag_info = _context.get_target_info(flag_target)
        if not flag_info.get(BuildSettingProvider):
          return False
        if str(flag_info[BuildSettingProvider].value) != value:
          return False
    if constraint_values:
      for constraint in _context.resolve_target_or_label_list(
          constraint_values
      ):
        if not _context.evaluate_condition(constraint):
          return False
    workspace_values = _context.access(EvaluationState).workspace.values
    if values:
      for key, value in values.items():
        if (key, value) not in workspace_values:
          return False
    if define_values:
      for key, value in define_values.items():
        if ("define", f"{key}={value}") not in workspace_values:
          return False
    return True

  evaluated_condition = evaluate()
  _context.add_analyzed_target(
      _target, TargetInfo(ConditionProvider(evaluated_condition))
  )
