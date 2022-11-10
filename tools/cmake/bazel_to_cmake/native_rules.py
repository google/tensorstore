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
"""CMake implementation of native Bazel build rules."""

# pylint: disable=relative-beyond-top-level,invalid-name,missing-function-docstring,g-long-lambda

import glob as _glob
import os
import pathlib
import re
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
from .util import glob_pattern_to_regexp
from .variable_substitution import apply_location_substitutions
from .variable_substitution import apply_make_variable_substitutions


@register_native_build_rule
def glob(self: EvaluationContext,
         include: List[str],
         exclude: Optional[List[str]] = None,
         allow_empty: bool = True) -> List[str]:
  matches = set()

  exclude_regexp = None
  if exclude:
    exclude_regexp = re.compile("(?:" + "|".join(
        glob_pattern_to_regexp(pattern) for pattern in exclude) + ")")

  assert self.current_package_name is not None
  package_directory = str(
      pathlib.PurePosixPath(self.repo.source_directory).joinpath(
          self.current_package_name))

  is_subpackage_dir: Dict[str, bool] = {}
  is_subpackage_dir[""] = False

  def in_subpackage(path: str):
    start_index = 1
    end_index = len(path)
    while start_index < end_index:
      index = path.find("/", start_index)
      if index == -1:
        break
      start_index = index + 1
      subdir = path[:index]
      result = is_subpackage_dir.get(subdir)
      if result is not None:
        return result
      build_path = str(
          pathlib.PurePosixPath(package_directory).joinpath(subdir, "BUILD"))
      result = (
          os.path.exists(build_path) or os.path.exists(build_path + ".bazel"))
      is_subpackage_dir[subdir] = result
      if result:
        return True
    return False

  def get_matches(pattern: str):
    for match in _glob.iglob(
        os.path.join(package_directory, pattern), recursive=True):
      if not os.path.isfile(match):
        continue
      relative = os.path.relpath(match, package_directory)
      if os.sep != "/":
        relative = relative.replace(os.sep, "/")
      if in_subpackage(relative):
        continue
      if exclude_regexp is not None and exclude_regexp.fullmatch(relative):
        continue
      yield relative

  for pattern in include:
    matches.update(get_matches(pattern))

  if not matches and not allow_empty:
    raise ValueError("glob produced empty result list")
  return sorted(matches)


@register_native_build_rule
def package_name(self: EvaluationContext):
  return self.current_package_name


@register_native_build_rule
def repository_name(self: EvaluationContext):
  return self.current_repository_name


@register_native_build_rule
def package_group(self: EvaluationContext, **kwargs):
  del self
  del kwargs
  pass


@register_native_build_rule
def filegroup(self: EvaluationContext, name: str, srcs: List[RelativeLabel],
              **kwargs):
  del kwargs
  package = self.current_package
  assert package is not None
  target = package.get_label(name)

  def impl():
    cmake_deps: List[CMakeTarget] = []
    providers = [
        FilesProvider(
            package.context.get_targets_file_paths(
                package.get_label_list(srcs), cmake_deps))
    ]
    if cmake_deps:
      providers.append(CMakeDepsProvider(cmake_deps))
    self.add_analyzed_target(target, TargetInfo(*providers))

  self.add_rule(target, impl, analyze_by_default=False)


@register_native_build_rule
def proto_library(self: EvaluationContext,
                  name: str,
                  visibility: Optional[List[RelativeLabel]] = None,
                  **kwargs):
  package = self.current_package
  assert package is not None
  label = package.get_label(name)
  self.add_rule(
      label,
      lambda: _proto_library_impl(cast(Package, package), label, **kwargs),
      analyze_by_default=package.analyze_by_default(visibility))


def _proto_library_impl(
    _package: Package,
    _label: Label,
    srcs: Optional[List[RelativeLabel]] = None,
    deps: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  del kwargs
  resolved_srcs = _package.get_label_list(srcs)
  resolved_deps = _package.get_label_list(deps)
  _package.context.add_analyzed_target(
      _label, TargetInfo(ProtoLibraryProvider(resolved_srcs, resolved_deps)))


@register_native_build_rule
def genrule(self: EvaluationContext,
            name: str,
            outs: List[RelativeLabel],
            visibility: Optional[List[RelativeLabel]] = None,
            **kwargs):
  package = self.current_package
  assert package is not None
  label = package.get_label(name)
  out_targets = [package.get_label(out) for out in outs]

  self.add_rule(
      label,
      lambda: _genrule_impl(
          cast(Package, package), label, out_targets, **kwargs),
      outs=out_targets,
      analyze_by_default=package.analyze_by_default(visibility))


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
  src_files = _package.context.get_targets_file_paths(
      _package.get_label_list(srcs), cmake_deps)
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
      toolchains=_package.get_label_list(toolchains))
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
  package = self.current_package
  assert package is not None
  if constraint_values:
    resolved_constraint_values = [
        package.get_label(condition) for condition in constraint_values
    ]
  if flag_values:
    resolved_flag_values = {
        package.get_label(flag): value for flag, value in flag_values.items()
    }

  def evaluate() -> bool:
    if flag_values:
      for flag, value in resolved_flag_values.items():
        if self.evaluate_build_setting(flag) != value:
          return False
    if constraint_values:
      for constraint in resolved_constraint_values:
        if not self.evaluate_condition(constraint):
          return False
    if values:
      for key, value in values.items():
        if (key, value) not in self.workspace.values:
          return False
    if define_values:
      for key, value in define_values.items():
        if ("define", f"{key}={value}") not in self.workspace.values:
          return False
    return True

  label = package.get_label(name)
  self.add_rule(
      label,
      lambda: self.add_analyzed_target(
          label, TargetInfo(ConditionProvider(evaluate()))),
      analyze_by_default=True)


@register_native_build_rule
def alias(self: EvaluationContext,
          name: str,
          actual: Configurable[RelativeLabel],
          visibility: Optional[List[RelativeLabel]] = None,
          **kwargs):
  del kwargs
  package = self.current_package
  assert package is not None
  label = package.get_label(name)

  def impl():
    self.add_analyzed_target(
        label,
        self.get_target_info(
            package.get_label(self.evaluate_configurable(actual))))

  self.add_rule(
      label, impl, analyze_by_default=package.analyze_by_default(visibility))


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
