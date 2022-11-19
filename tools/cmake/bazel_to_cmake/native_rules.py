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

import os
import pathlib
from typing import Dict, List, Optional, cast

from . import cmake_builder
from .cmake_builder import CMakeBuilder
from .configurable import Configurable
from .evaluation import EvaluationContext
from .evaluation import Package
from .evaluation import register_native_build_rule
from .label import CMakeTarget
from .label import Label
from .label import RelativeLabel
from .provider import CMakeDepsProvider
from .provider import ConditionProvider
from .provider import FilesProvider
from .provider import ProtoLibraryProvider
from .provider import TargetInfo
from .starlark.bazel_glob import glob as starlark_glob
from .variable_substitution import apply_location_substitutions
from .variable_substitution import apply_make_variable_substitutions


@register_native_build_rule
def package(self: EvaluationContext, **kwargs):
  default_visibility = kwargs.get("default_visibility")
  if default_visibility:
    _package = self.current_package
    assert _package is not None
    _package.default_visibility = _package.get_label_list(default_visibility)


@register_native_build_rule
def package_name(self: EvaluationContext):
  return self.current_package_name


@register_native_build_rule
def package_group(self: EvaluationContext, **kwargs):
  del self
  del kwargs
  pass


@register_native_build_rule
def repository_name(self: EvaluationContext):
  return self.current_repository_name


@register_native_build_rule
def existing_rule(self: EvaluationContext, name):
  _package = self.current_package
  assert _package is not None
  label = _package.get_label(name)
  r = self._all_rules.get(name, None)
  if r:
    return r
  return self._all_rules.get(label, None)


@register_native_build_rule
def glob(self: EvaluationContext,
         include: List[str],
         exclude: Optional[List[str]] = None,
         allow_empty: bool = True) -> List[str]:

  assert self.current_package_name is not None
  package_directory = str(
      pathlib.PurePosixPath(self.repo.source_directory).joinpath(
          self.current_package_name))

  return starlark_glob(package_directory, include, exclude, allow_empty)


@register_native_build_rule
def filegroup(self: EvaluationContext, name: str, srcs: List[RelativeLabel],
              **kwargs):
  del kwargs
  _package = self.current_package
  assert _package is not None
  target = _package.get_label(name)

  def impl():
    resolved_srcs = _package.get_label_list(
        _package.get_configurable_list(srcs))
    cmake_deps: List[CMakeTarget] = []
    providers = [
        FilesProvider(
            _package.context.get_targets_file_paths(resolved_srcs, cmake_deps))
    ]
    if cmake_deps:
      providers.append(CMakeDepsProvider(cmake_deps))
    self.add_analyzed_target(target, TargetInfo(*providers))

  self.add_rule(target, impl, analyze_by_default=False)


@register_native_build_rule
def proto_library(self: EvaluationContext,
                  name: str,
                  srcs: Optional[List[RelativeLabel]],
                  deps: Optional[List[RelativeLabel]] = None,
                  visibility: Optional[List[RelativeLabel]] = None,
                  **kwargs):
  del kwargs
  _package = self.current_package
  assert _package is not None
  label = _package.get_label(name)

  def impl():
    resolved_srcs = _package.get_label_list(
        _package.get_configurable_list(srcs))
    resolved_deps = _package.get_label_list(
        _package.get_configurable_list(deps))
    _package.context.add_analyzed_target(
        label, TargetInfo(ProtoLibraryProvider(resolved_srcs, resolved_deps)))

  self.add_rule(
      label, impl, analyze_by_default=_package.analyze_by_default(visibility))


@register_native_build_rule
def genrule(self: EvaluationContext,
            name: str,
            outs: List[RelativeLabel],
            visibility: Optional[List[RelativeLabel]] = None,
            **kwargs):
  _package = self.current_package
  assert _package is not None
  label = _package.get_label(name)
  out_targets = [_package.get_label(out) for out in outs]

  self.add_rule(
      label,
      lambda: _genrule_impl(
          cast(Package, _package), label, out_targets, **kwargs),
      outs=out_targets,
      analyze_by_default=_package.analyze_by_default(visibility))


def _get_relative_path(path: str, relative_to: str) -> str:
  rel_path = os.path.relpath(path, relative_to)
  if os.sep != "/":
    rel_path = rel_path.replace(os.sep, "/")
  return rel_path


def _genrule_impl(_package: Package,
                  _label: Label,
                  _out_targets: List[Label],
                  cmd: Configurable[str],
                  srcs: Optional[Configurable[List[RelativeLabel]]] = None,
                  message: Optional[Configurable[str]] = None,
                  toolchains: Optional[List[RelativeLabel]] = None,
                  **kwargs):
  # Resolve srcs & toolchains
  resolved_srcs = _package.get_label_list(_package.get_configurable_list(srcs))
  resolved_toolchains = _package.get_label_list(
      _package.get_configurable_list(toolchains))

  context = _package.context
  cmake_target_pair = context.generate_cmake_target_pair(
      _label, generate_alias=False)
  cmake_deps_provider = CMakeDepsProvider([cmake_target_pair.target])
  out_files: List[str] = []
  for out_target in _out_targets:
    out_file = context.get_generated_file_path(out_target)
    out_files.append(out_file)
    context.add_analyzed_target(
        out_target, TargetInfo(FilesProvider([out_file]), cmake_deps_provider))
  del kwargs
  message_text = ""
  if message is not None:
    message_text = _package.context.evaluate_configurable(message)
  cmake_deps: List[CMakeTarget] = []
  src_files = _package.context.get_targets_file_paths(resolved_srcs, cmake_deps)
  cmake_deps.extend(src_files)
  relative_source_paths = [
      _get_relative_path(path, _package.repo.source_directory)
      for path in src_files
  ]
  relative_out_paths = [
      _get_relative_path(path, _package.repo.source_directory)
      for path in out_files
  ]
  substitutions = {
      "$(SRCS)": " ".join(relative_source_paths),
      "$(OUTS)": " ".join(relative_out_paths),
  }
  if len(out_files) == 1:
    substitutions["$@"] = relative_out_paths[0]
  if len(src_files) == 1:
    substitutions["$<"] = relative_source_paths[0]
  # TODO(jbms): Add missing variables, including:
  #   "$(BINDIR)"
  #   "$(GENDIR)"
  #   "$(COMPILATION_MODE)"
  #   "$(TARGET_CPU)"
  #   "$(RULEDIR)"
  #   "$(@D)"
  cmd_text = _package.context.evaluate_configurable(cmd)
  cmd_text = apply_location_substitutions(
      _package, cmd_text, relative_to=_package.repo.source_directory)

  cmd_text = apply_make_variable_substitutions(
      _package.context.builder,
      cmd_text,
      substitutions=substitutions,
      toolchains=resolved_toolchains)
  _emit_genrule(
      builder=_package.context.builder,
      cmake_target=cmake_deps_provider.targets[0],
      out_files=out_files,
      cmake_deps=cmake_deps,
      cmd_text=cmd_text,
      message=message_text)
  context.add_analyzed_target(_label, TargetInfo(cmake_deps_provider))


def _emit_genrule(
    builder: CMakeBuilder,
    cmake_target: str,
    out_files: List[str],
    cmake_deps: List[str],
    cmd_text: str,
    message: Optional[str] = None,
):
  if message:
    optional_message_text = f"COMMENT {cmake_builder.quote_string(message)}"
  else:
    optional_message_text = ""
  builder.addtext(f"""
add_custom_command(
  OUTPUT {cmake_builder.quote_list(out_files)}
  DEPENDS {cmake_builder.quote_list(cmake_deps)}
  COMMAND {cmd_text}
  {optional_message_text}
  VERBATIM
  WORKING_DIRECTORY "${{CMAKE_CURRENT_SOURCE_DIR}}"
)
add_custom_target({cmake_target} DEPENDS {cmake_builder.quote_list(out_files)})
""")


@register_native_build_rule
def config_setting(
    self: EvaluationContext,
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
  _package = self.current_package
  assert _package is not None
  label = _package.get_label(name)

  def impl():
    _context = _package.context

    def evaluate() -> bool:
      if flag_values:
        for flag, value in flag_values.items():
          if _context.evaluate_build_setting(_package.get_label(flag)) != value:
            return False
      if constraint_values:
        for constraint in _package.get_label_list(constraint_values):
          if not _context.evaluate_condition(constraint):
            return False
      if values:
        for key, value in values.items():
          if (key, value) not in _context.workspace.values:
            return False
      if define_values:
        for key, value in define_values.items():
          if ("define", f"{key}={value}") not in _context.workspace.values:
            return False
      return True

    evaluated_condition = evaluate()
    _context.add_analyzed_target(
        label, TargetInfo(ConditionProvider(evaluated_condition)))

  self.add_rule(label, impl, analyze_by_default=True)


@register_native_build_rule
def alias(self: EvaluationContext,
          name: str,
          actual: Configurable[RelativeLabel],
          **kwargs):
  # Aliases are not analyzed by default.
  del kwargs

  _package = self.current_package
  assert _package is not None
  label = _package.get_label(name)

  def impl():
    resolved = _package.get_label(_package.get_configurable(actual))
    _package.context.add_analyzed_target(
        label, _package.context.get_target_info(resolved))

  self.add_rule(label, impl, analyze_by_default=False)


@register_native_build_rule
def exports_files(self: EvaluationContext, *args, **kwargs):
  del self
  del args
  del kwargs


@register_native_build_rule
def py_library(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def py_test(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def py_binary(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def py_proto_library(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def java_library(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def java_test(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def java_binary(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def java_proto_library(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def go_library(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def go_test(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def go_binary(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def go_proto_library(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def objc_library(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def objc_test(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def objc_binary(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def objc_proto_library(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def sh_binary(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def sh_test(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs
