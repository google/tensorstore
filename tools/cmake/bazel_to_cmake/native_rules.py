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

from .cmake_target import CMakeDepsProvider
from .cmake_target import CMakeTarget
from .evaluation import EvaluationState
from .package import Visibility
from .starlark import rule  # pylint: disable=unused-import
from .starlark.bazel_glob import glob as starlark_glob
from .starlark.bazel_globals import register_native_build_rule
from .starlark.common_providers import ConditionProvider
from .starlark.common_providers import FilesProvider
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.provider import Provider
from .starlark.provider import TargetInfo


@register_native_build_rule
def repository_name(self: InvocationContext):
  return f"@{self.caller_package_id.repository_name}"


@register_native_build_rule
def package_name(self: InvocationContext):
  return self.caller_package_id.package_name


@register_native_build_rule
def package(self: InvocationContext,
            default_visibility: Optional[List[RelativeLabel]] = None,
            **kwargs):
  del kwargs
  if default_visibility:
    self.access(Visibility).set_default_visibility(
        self.resolve_target_or_label_list(default_visibility))


@register_native_build_rule
def package_group(self: InvocationContext, **kwargs):
  del self
  del kwargs
  pass


@register_native_build_rule
def existing_rule(self: InvocationContext, name: str):
  target = self.resolve_target(name)
  return self.access(EvaluationState)._all_rules.get(target, None)  # pylint: disable=protected-access


@register_native_build_rule
def glob(self: InvocationContext,
         include: List[str],
         exclude: Optional[List[str]] = None,
         allow_empty: bool = True) -> List[str]:

  package_directory = self.get_source_package_dir(self.caller_package_id)
  return starlark_glob(package_directory, include, exclude, allow_empty)


@register_native_build_rule
def filegroup(self: InvocationContext,
              name: str,
              srcs: Optional[List[RelativeLabel]] = None,
              **kwargs):
  # https://bazel.build/reference/be/general#filegroup
  del kwargs
  _context = self.snapshot()
  target = _context.resolve_target(name)

  def impl() -> None:
    resolved_srcs = _context.resolve_target_or_label_list(
        _context.evaluate_configurable_list(srcs))
    cmake_deps: List[CMakeTarget] = []

    state = _context.access(EvaluationState)
    providers: List[Provider] = [
        FilesProvider(state.get_targets_file_paths(resolved_srcs, cmake_deps))
    ]
    if cmake_deps:
      providers.append(CMakeDepsProvider(cmake_deps))
    _context.add_analyzed_target(target, TargetInfo(*providers))

  _context.add_rule(target, impl, analyze_by_default=False)


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
  _context = self.snapshot()
  target = _context.resolve_target(name)

  def impl():

    def evaluate() -> bool:
      if flag_values:
        for flag, value in flag_values.items():
          if _context.evaluate_build_setting(
              _context.resolve_target_or_label(flag)) != value:
            return False
      if constraint_values:
        for constraint in _context.resolve_target_or_label_list(
            constraint_values):
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
        target, TargetInfo(ConditionProvider(evaluated_condition)))

  _context.add_rule(target, impl, analyze_by_default=True)


@register_native_build_rule
def exports_files(self: InvocationContext, *args, **kwargs):
  del self
  del args
  del kwargs


@register_native_build_rule
def py_library(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def py_test(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def py_binary(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def py_proto_library(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def java_library(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def java_test(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def java_binary(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def java_proto_library(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def go_library(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def go_test(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def go_binary(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def go_proto_library(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def objc_library(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def objc_test(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def objc_binary(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def objc_proto_library(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def sh_binary(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def sh_test(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs
