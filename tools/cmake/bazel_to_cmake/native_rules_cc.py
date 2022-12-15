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
"""CMake implementation of native Bazel build cc_* rules."""

# pylint: disable=relative-beyond-top-level,invalid-name,missing-function-docstring,g-long-lambda

from typing import List, Optional

from .cmake_builder import CMakeBuilder
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetProvider
from .emit_cc import emit_cc_binary
from .emit_cc import emit_cc_library
from .emit_cc import emit_cc_test
from .emit_cc import handle_cc_common_options
from .evaluation import EvaluationState
from .package import Visibility
from .starlark.bazel_globals import register_native_build_rule
from .starlark.bazel_target import TargetId
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.provider import TargetInfo
from .starlark.select import Configurable
from .variable_substitution import apply_location_substitutions


@register_native_build_rule
def cc_library(self: InvocationContext,
               name: str,
               visibility: Optional[List[RelativeLabel]] = None,
               **kwargs):
  context = self.snapshot()
  target = context.resolve_target(name)
  context.add_rule(
      target,
      lambda: _cc_library_impl(context, target, **kwargs),
      visibility=visibility)


def _cc_library_impl(
    _context: InvocationContext,
    _target: TargetId,
    hdrs: Optional[Configurable[List[RelativeLabel]]] = None,
    textual_hdrs: Optional[Configurable[List[RelativeLabel]]] = None,
    alwayslink: bool = False,
    **kwargs,
):
  resolved_hdrs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(hdrs))
  resolved_textual_hdrs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(textual_hdrs))

  state = _context.access(EvaluationState)

  cmake_target_pair = state.generate_cmake_target_pair(_target)
  custom_target_deps: List[CMakeTarget] = []
  hdrs_file_paths = state.get_targets_file_paths(
      resolved_hdrs, custom_target_deps=custom_target_deps)
  textual_hdrs_file_paths = state.get_targets_file_paths(
      resolved_textual_hdrs, custom_target_deps=custom_target_deps)

  common_options = handle_cc_common_options(
      _context, custom_target_deps=custom_target_deps, **kwargs)
  builder = _context.access(CMakeBuilder)
  builder.addtext(f"\n# {_target.as_label()}")

  emit_cc_library(
      builder,
      cmake_target_pair,
      hdrs=set(hdrs_file_paths + textual_hdrs_file_paths),
      alwayslink=alwayslink,
      **common_options,
  )
  _context.add_analyzed_target(_target,
                               TargetInfo(*cmake_target_pair.as_providers()))


@register_native_build_rule
def cc_binary(self: InvocationContext,
              name: str,
              visibility: Optional[List[RelativeLabel]] = None,
              **kwargs):
  context = self.snapshot()
  target = context.resolve_target(name)

  resolved_visibility = context.resolve_target_or_label_list(visibility or [])
  if kwargs.get("testonly"):
    analyze_by_default = context.access(Visibility).analyze_test_by_default(
        resolved_visibility)
  else:
    analyze_by_default = context.access(Visibility).analyze_by_default(
        resolved_visibility)

  context.add_rule(
      target,
      lambda: _cc_binary_impl(context, target, **kwargs),
      analyze_by_default=analyze_by_default)


def _cc_binary_impl(_context: InvocationContext, _target: TargetId, **kwargs):
  cmake_target_pair = _context.access(
      EvaluationState).generate_cmake_target_pair(_target)

  common_options = handle_cc_common_options(
      _context, src_required=True, **kwargs)
  builder = _context.access(CMakeBuilder)
  builder.addtext(f"\n# {_target.as_label()}")

  emit_cc_binary(
      _context.access(CMakeBuilder),
      cmake_target_pair,
      **common_options,
  )
  _context.add_analyzed_target(_target,
                               TargetInfo(*cmake_target_pair.as_providers()))


@register_native_build_rule
def cc_test(self: InvocationContext,
            name: str,
            visibility: Optional[List[RelativeLabel]] = None,
            **kwargs):
  context = self.snapshot()
  target = context.resolve_target(name)

  resolved_visibility = context.resolve_target_or_label_list(visibility or [])
  context.add_rule(
      target,
      lambda: _cc_test_impl(context, target, **kwargs),
      analyze_by_default=context.access(Visibility).analyze_test_by_default(
          resolved_visibility))


def _cc_test_impl(_context: InvocationContext,
                  _target: TargetId,
                  args: Optional[Configurable[List[str]]] = None,
                  **kwargs):
  state = _context.access(EvaluationState)
  cmake_target_pair = state.generate_cmake_target_pair(_target)
  resolved_args = [
      apply_location_substitutions(
          _context, arg, relative_to=state.repo.source_directory)
      for arg in _context.evaluate_configurable_list(args)
  ]

  common_options = handle_cc_common_options(
      _context, src_required=True, **kwargs)
  builder = _context.access(CMakeBuilder)
  builder.addtext(f"\n# {_target.as_label()}")

  emit_cc_test(
      _context.access(CMakeBuilder),
      cmake_target_pair,
      args=resolved_args,
      **common_options,
  )
  _context.add_analyzed_target(_target,
                               TargetInfo(*cmake_target_pair.as_providers()))


@register_native_build_rule
def alias(self: InvocationContext,
          name: str,
          actual: Configurable[RelativeLabel],
          visibility: Optional[List[RelativeLabel]] = None,
          **kwargs):
  del kwargs
  context = self.snapshot()
  target = context.resolve_target(name)
  context.add_rule(
      target,
      lambda: _alias_impl(context, target, actual),
      visibility=visibility)


def _alias_impl(_context: InvocationContext, _target: TargetId,
                actual: Configurable[RelativeLabel]):
  resolved = _context.resolve_target(_context.evaluate_configurable(actual))
  target_info = _context.get_target_info(resolved)
  _context.add_analyzed_target(_target, target_info)

  if resolved.package_id != _context.caller_package_id:
    # When this is an alias to another package, don't add a CMake ALIAS.
    return

  if target_info.get(CMakeTargetProvider) is None:
    # When there is no CMake target, don't add a CMake ALIAS.
    # NOTE: We might want to alias proto_library().
    return

  alias_target = target_info[CMakeTargetProvider].target
  cmake_target_pair = _context.access(
      EvaluationState).generate_cmake_target_pair(_target)

  if cmake_target_pair.target == alias_target:
    # Don't alias, when, unexpectedly, the targets have the same name.
    return

  builder = _context.access(CMakeBuilder)
  builder.addtext(f"\n# {_target.as_label()}\n")
  builder.addtext(
      f"add_library({cmake_target_pair.target} ALIAS {alias_target})\n")
  if cmake_target_pair.alias:
    builder.addtext(
        f"add_library({cmake_target_pair.alias} ALIAS {alias_target})\n")
