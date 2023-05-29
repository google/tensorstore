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
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetPair
from .cmake_target import CMakeTargetProvider
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
  target = context.resolve_target(name)
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

  alias_target = target_info.get(CMakeTargetProvider)
  if alias_target is None or alias_target.target is None:
    # When there is no CMake target, don't add a CMake ALIAS.
    # NOTE: We might want to alias proto_library().
    return

  state = _context.access(EvaluationState)
  cmake_target_pair = state.generate_cmake_target_pair(_target)

  if cmake_target_pair.target == alias_target.target:
    # Don't alias, when, unexpectedly, the targets have the same name.
    return

  _emit_cmake_alias(
      _context,
      f"\n# {_target.as_label()}\n",
      cmake_target_pair,
      alias_target.target,
  )


def _emit_cmake_alias(
    _context: InvocationContext,
    prefix_str: str,
    source: CMakeTargetPair,
    dest: CMakeTarget,
):
  builder = _context.access(CMakeBuilder)
  if prefix_str:
    builder.addtext(prefix_str)
  builder.addtext(f"add_library({source.target} ALIAS {dest})\n")
  if source.alias:
    builder.addtext(f"add_library({source.alias} ALIAS {dest})\n")
