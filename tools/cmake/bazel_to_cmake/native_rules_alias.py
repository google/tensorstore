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

from typing import List, Optional

from .cmake_builder import CMakeBuilder
from .cmake_target import CMakeExecutableTargetProvider
from .cmake_target import CMakeLibraryTargetProvider
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetPair
from .evaluation import EvaluationState
from .starlark.bazel_globals import register_native_build_rule
from .starlark.bazel_target import TargetId
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
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
  resolved = _context.resolve_target_or_label(
      _context.evaluate_configurable(actual)
  )
  target_info = _context.get_target_info(resolved)
  _context.add_analyzed_target(_target, target_info)

  if resolved.repository_name != _context.caller_package_id.repository_name:
    # When this is an alias to another repository, don't add a CMake ALIAS.
    return

  state = _context.access(EvaluationState)

  resolved_provider = target_info.get(CMakeLibraryTargetProvider)
  if resolved_provider is None:
    resolved_provider = target_info.get(CMakeExecutableTargetProvider)
  if resolved_provider is not None:
    # When the rule resolves to a CMakeTarget, emit an alias for that target.
    _emit_cmake_alias(
        _context,
        f"\n# alias({_target.as_label()})\n",
        state.generate_cmake_target_pair(_target),
        resolved_provider.target,
        is_executable=isinstance(
            resolved_provider, CMakeExecutableTargetProvider
        ),
    )


def _emit_cmake_alias(
    _context: InvocationContext,
    prefix_str: str,
    source: CMakeTargetPair,
    dest: CMakeTarget,
    is_executable: bool,
):
  # Don't emit an alias for self.
  if source.target == dest:
    return
  add_fn = ["add_library", "add_executable"][is_executable]
  builder = _context.access(CMakeBuilder)
  if prefix_str:
    builder.addtext(prefix_str)
  builder.addtext(f"{add_fn}({source.target} ALIAS {dest})\n")
  if source.alias:
    builder.addtext(f"{add_fn}({source.alias} ALIAS {dest})\n")
