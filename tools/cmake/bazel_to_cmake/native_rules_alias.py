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
"""CMake implementation of native Bazel alias() rule."""

# pylint: disable=relative-beyond-top-level,invalid-name,missing-function-docstring,g-long-lambda
import io
from typing import List, Optional

from .cmake_builder import CMakeBuilder
from .cmake_provider import CMakeAddDependenciesProvider
from .cmake_provider import CMakeAliasProvider
from .cmake_provider import CMakeExecutableTargetProvider
from .cmake_provider import CMakeLinkLibrariesProvider
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetPair
from .evaluation import EvaluationState
from .starlark.bazel_build_file import register_native_build_rule
from .starlark.bazel_target import TargetId
from .starlark.common_providers import BuildSettingProvider
from .starlark.common_providers import ConditionProvider
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.provider import TargetInfo
from .starlark.select import Configurable


@register_native_build_rule
def alias(
    self: InvocationContext,
    name: str,
    actual: Configurable[RelativeLabel],
    visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  del kwargs
  context = self.snapshot()
  target = context.parse_rule_target(name)
  context.add_rule(
      target,
      lambda: _alias_impl(context, target, actual),
      visibility=visibility,
  )


def _alias_impl(
    _context: InvocationContext,
    _target: TargetId,
    actual: Configurable[RelativeLabel],
):
  state = _context.access(EvaluationState)

  resolved = _context.resolve_target_or_label(
      _context.evaluate_configurable(actual)
  )
  target_info = _context.get_target_info(resolved)

  # Special case any target info with conditions or build settings.
  if target_info.get(ConditionProvider) or target_info.get(
      BuildSettingProvider
  ):
    _context.add_analyzed_target(_target, target_info)
    return

  # CMake doesn't allow aliases to aliases, so if that happens, use the
  # same alias target.
  if target_info.get(CMakeAliasProvider):
    cmake_alias_target = target_info[CMakeAliasProvider].target
  else:
    cmake_alias_target = state.generate_cmake_target_pair(resolved).target

  _context.add_analyzed_target(
      _target,
      TargetInfo(*tuple(target_info), CMakeAliasProvider(cmake_alias_target)),
  )

  if resolved.repository_name != _context.caller_package_id.repository_name:
    # When this is an alias to another repository, don't add a CMake ALIAS.
    return

  source = state.generate_cmake_target_pair(_target)
  if source.target == cmake_alias_target:
    return

  function = None
  if target_info.get(CMakeExecutableTargetProvider):
    function = "add_executable"
  elif target_info.get(CMakeLinkLibrariesProvider) or target_info.get(
      CMakeAddDependenciesProvider
  ):
    function = "add_library"

  out = io.StringIO()
  out.write(f"\n# alias({_target.as_label()})\n")
  if function:
    out.write(f"{function}({source.target} ALIAS {cmake_alias_target})\n")
    if source.alias:
      out.write(f"{function}({source.alias} ALIAS {cmake_alias_target})\n")
  else:
    out.write(f"# No alias emitted for {cmake_alias_target})\n")
  _context.access(CMakeBuilder).addtext(out.getvalue())
