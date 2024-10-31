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

import io
import itertools
from typing import Callable, Dict, List, Optional

from .cmake_builder import CMakeBuilder
from .cmake_provider import CMakeExecutableTargetProvider
from .cmake_provider import CMakePackageDepsProvider
from .cmake_provider import default_providers
from .cmake_provider import make_providers
from .cmake_target import CMakeTarget
from .emit_cc import emit_cc_binary
from .emit_cc import emit_cc_library
from .emit_cc import emit_cc_test
from .emit_cc import handle_cc_common_options
from .evaluation import EvaluationState
from .package import Visibility
from .starlark.bazel_build_file import register_native_build_rule
from .starlark.bazel_target import TargetId
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.provider import TargetInfo
from .starlark.select import Configurable
from .variable_substitution import apply_location_substitutions


def _common_cc_resolve(
    _next: Callable[..., None],
    _context: InvocationContext,
    _target: TargetId,
    srcs: Optional[Configurable[List[RelativeLabel]]] = None,
    hdrs: Optional[Configurable[List[RelativeLabel]]] = None,
    textual_hdrs: Optional[Configurable[List[RelativeLabel]]] = None,
    deps: Optional[Configurable[List[RelativeLabel]]] = None,
    implementation_deps: Optional[Configurable[List[RelativeLabel]]] = None,
    includes: Optional[Configurable[List[str]]] = None,
    include_prefix: Optional[Configurable[str]] = None,
    strip_include_prefix: Optional[Configurable[str]] = None,
    tags: Optional[Configurable[List[str]]] = None,
    args: Optional[Configurable[List[str]]] = None,
    **kwargs,
):
  """Applies evaluate_configurable to common (and uncommon) cc rule arguments."""
  _next(
      _context,
      _target,
      srcs=_context.resolve_target_or_label_list(
          _context.evaluate_configurable_list(srcs)
      ),
      hdrs=_context.resolve_target_or_label_list(
          _context.evaluate_configurable_list(hdrs)
      ),
      deps=_context.resolve_target_or_label_list(
          _context.evaluate_configurable_list(deps)
      ),
      implementation_deps=_context.resolve_target_or_label_list(
          _context.evaluate_configurable_list(implementation_deps)
      ),
      textual_hdrs=_context.resolve_target_or_label_list(
          _context.evaluate_configurable_list(textual_hdrs)
      ),
      includes=_context.evaluate_configurable_list(includes),
      include_prefix=(
          _context.evaluate_configurable(include_prefix)
          if include_prefix
          else None
      ),
      strip_include_prefix=(
          _context.evaluate_configurable(strip_include_prefix)
          if strip_include_prefix
          else None
      ),
      tags=_context.evaluate_configurable_list(tags),
      args=_context.evaluate_configurable_list(args),
      **kwargs,
  )


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
      lambda: _common_cc_resolve(_cc_library_impl, context, target, **kwargs),
      visibility=visibility,
  )


def _cc_library_impl(
    _context: InvocationContext,
    _target: TargetId,
    hdrs: Optional[List[TargetId]] = None,
    textual_hdrs: Optional[List[TargetId]] = None,
    alwayslink: bool = False,
    **kwargs,
):
  state = _context.access(EvaluationState)

  hdrs_collector = state.collect_targets(hdrs)
  textual_hdrs_collector = state.collect_targets(textual_hdrs)

  add_dependencies: List[CMakeTarget] = list(
      itertools.chain(
          hdrs_collector.add_dependencies(),
          textual_hdrs_collector.add_dependencies(),
      )
  )
  all_hdrs = set(
      itertools.chain(
          hdrs_collector.file_paths(), textual_hdrs_collector.file_paths()
      )
  )

  cmake_target_pair = state.generate_cmake_target_pair(_target)

  common_options = handle_cc_common_options(
      _context,
      add_dependencies=add_dependencies,
      hdrs_file_paths=list(hdrs_collector.file_paths()),
      textual_hdrs_file_paths=list(textual_hdrs_collector.file_paths()),
      **kwargs,
  )

  out = io.StringIO()
  out.write(f"\n# cc_library({_target.as_label()})")
  extra_providers = emit_cc_library(
      out,
      cmake_target_pair,
      hdrs=all_hdrs,
      alwayslink=alwayslink,
      **common_options,
  )
  _context.access(CMakeBuilder).addtext(out.getvalue())
  _context.add_analyzed_target(
      _target,
      TargetInfo(*default_providers(cmake_target_pair), *extra_providers),
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
      lambda: _common_cc_resolve(_cc_binary_impl, context, target, **kwargs),
      analyze_by_default=analyze_by_default,
  )


def _cc_binary_impl(_context: InvocationContext, _target: TargetId, **kwargs):
  cmake_target_pair = _context.access(
      EvaluationState
  ).generate_cmake_target_pair(_target)

  common_options = handle_cc_common_options(
      _context, src_required=True, **kwargs
  )

  out = io.StringIO()
  out.write(f"\n# cc_binary({_target.as_label()})")
  emit_cc_binary(
      out,
      cmake_target_pair,
      **common_options,
  )
  _context.access(CMakeBuilder).addtext(out.getvalue())
  _context.add_analyzed_target(
      _target,
      TargetInfo(
          *make_providers(
              cmake_target_pair,
              CMakePackageDepsProvider,
              CMakeExecutableTargetProvider,
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
  context = self.snapshot()
  target = context.parse_rule_target(name)

  resolved_visibility = context.resolve_target_or_label_list(visibility or [])
  context.add_rule(
      target,
      lambda: _common_cc_resolve(_cc_test_impl, context, target, **kwargs),
      analyze_by_default=context.access(Visibility).analyze_test_by_default(
          resolved_visibility
      ),
  )


def _cc_test_impl(
    _context: InvocationContext,
    _target: TargetId,
    args: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    **kwargs,
):
  # CMake does not run tests multiple times, so skip the flaky tests.
  if "skip-cmake" in tags or "flaky" in tags:
    _context.access(CMakeBuilder).addtext(
        f"\n# Skipping cc_test {_target.as_label()}\n"
    )
    _context.add_analyzed_target(_target, TargetInfo())
    return

  state = _context.access(EvaluationState)

  cmake_target_pair = state.generate_cmake_target_pair(_target)
  resolved_args = [
      apply_location_substitutions(
          _context, arg, relative_to=state.active_repo.source_directory
      )
      for arg in args
  ]

  common_options = handle_cc_common_options(
      _context, src_required=True, **kwargs
  )
  # Translate tags to CMake properties.
  # https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html#test-properties
  properties: Dict[str, str] = {}
  for t in tags:
    if t.startswith("cpu:"):
      properties["RUN_SERIAL"] = "TRUE"

  out = io.StringIO()
  out.write(f"\n# cc_test({_target.as_label()})")
  emit_cc_test(
      out,
      cmake_target_pair,
      args=resolved_args,
      properties=properties,
      **common_options,
  )
  _context.access(CMakeBuilder).addtext(out.getvalue())
  _context.add_analyzed_target(
      _target,
      TargetInfo(
          *make_providers(
              cmake_target_pair,
              CMakePackageDepsProvider,
              CMakeExecutableTargetProvider,
          )
      ),
  )
