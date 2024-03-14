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
from .cmake_target import CMakeExecutableTargetProvider
from .cmake_target import CMakeTarget
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
def cc_library(
    self: InvocationContext,
    name: str,
    visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  if "skip-cmake" in kwargs.get("tags", []):
    return
  context = self.snapshot()

  target = context.parse_rule_target(name)
  context.add_rule(
      target,
      lambda: _cc_library_impl(context, target, **kwargs),
      visibility=visibility,
  )


def _cc_library_impl(
    _context: InvocationContext,
    _target: TargetId,
    hdrs: Optional[Configurable[List[RelativeLabel]]] = None,
    textual_hdrs: Optional[Configurable[List[RelativeLabel]]] = None,
    alwayslink: bool = False,
    **kwargs,
):
  resolved_hdrs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(hdrs)
  )
  resolved_textual_hdrs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(textual_hdrs)
  )

  state = _context.access(EvaluationState)

  cmake_target_pair = state.generate_cmake_target_pair(_target)
  custom_target_deps: List[CMakeTarget] = []
  hdrs_file_paths = state.get_targets_file_paths(
      resolved_hdrs, custom_target_deps=custom_target_deps
  )
  textual_hdrs_file_paths = state.get_targets_file_paths(
      resolved_textual_hdrs, custom_target_deps=custom_target_deps
  )

  common_options = handle_cc_common_options(
      _context,
      custom_target_deps=custom_target_deps,
      hdrs_file_paths=hdrs_file_paths,
      textual_hdrs_file_paths=textual_hdrs_file_paths,
      **kwargs,
  )
  builder = _context.access(CMakeBuilder)
  builder.addtext(f"\n# cc_library({_target.as_label()})")

  emit_cc_library(
      builder,
      cmake_target_pair,
      hdrs=set(hdrs_file_paths + textual_hdrs_file_paths),
      alwayslink=alwayslink,
      **common_options,
  )
  _context.add_analyzed_target(
      _target, TargetInfo(*cmake_target_pair.as_providers())
  )


@register_native_build_rule
def cc_binary(
    self: InvocationContext,
    name: str,
    visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  if "skip-cmake" in kwargs.get("tags", []):
    return
  context = self.snapshot()
  target = context.parse_rule_target(name)

  resolved_visibility = context.resolve_target_or_label_list(visibility or [])
  if kwargs.get("testonly"):
    analyze_by_default = context.access(Visibility).analyze_test_by_default(
        resolved_visibility
    )
  else:
    analyze_by_default = context.access(Visibility).analyze_by_default(
        resolved_visibility
    )

  context.add_rule(
      target,
      lambda: _cc_binary_impl(context, target, **kwargs),
      analyze_by_default=analyze_by_default,
  )


def _cc_binary_impl(_context: InvocationContext, _target: TargetId, **kwargs):
  cmake_target_pair = _context.access(
      EvaluationState
  ).generate_cmake_target_pair(_target)

  common_options = handle_cc_common_options(
      _context, src_required=True, **kwargs
  )
  builder = _context.access(CMakeBuilder)
  builder.addtext(f"\n# cc_binary({_target.as_label()})")

  emit_cc_binary(
      _context.access(CMakeBuilder),
      cmake_target_pair,
      **common_options,
  )
  _context.add_analyzed_target(
      _target,
      TargetInfo(
          *cmake_target_pair.as_providers(
              provider=CMakeExecutableTargetProvider
          )
      ),
  )


@register_native_build_rule
def cc_test(
    self: InvocationContext,
    name: str,
    visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  if "skip-cmake" in kwargs.get("tags", []):
    return
  # CMake does not run tests multiple times, so skip the flaky tests.
  if kwargs.get("flaky", False):
    return
  context = self.snapshot()
  target = context.parse_rule_target(name)

  resolved_visibility = context.resolve_target_or_label_list(visibility or [])
  context.add_rule(
      target,
      lambda: _cc_test_impl(context, target, **kwargs),
      analyze_by_default=context.access(Visibility).analyze_test_by_default(
          resolved_visibility
      ),
  )


def _cc_test_impl(
    _context: InvocationContext,
    _target: TargetId,
    args: Optional[Configurable[List[str]]] = None,
    **kwargs,
):
  state = _context.access(EvaluationState)
  cmake_target_pair = state.generate_cmake_target_pair(_target)
  resolved_args = [
      apply_location_substitutions(
          _context, arg, relative_to=state.active_repo.source_directory
      )
      for arg in _context.evaluate_configurable_list(args)
  ]

  common_options = handle_cc_common_options(
      _context, src_required=True, **kwargs
  )
  builder = _context.access(CMakeBuilder)
  builder.addtext(f"\n# cc_test({_target.as_label()})")

  emit_cc_test(
      _context.access(CMakeBuilder),
      cmake_target_pair,
      args=resolved_args,
      **common_options,
  )
  _context.add_analyzed_target(
      _target,
      TargetInfo(
          *cmake_target_pair.as_providers(
              provider=CMakeExecutableTargetProvider
          )
      ),
  )
