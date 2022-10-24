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
"""CMake implementation of "@bazel_skylib"."""

# pylint: disable=invalid-name,relative-beyond-top-level,missing-function-docstring,missing-class-docstring,g-long-lambda

import json
import os
from typing import List, Dict, Optional, cast, Any

from .. import cmake_builder
from .. import native_rules
from ..configurable import Configurable
from ..evaluation import BazelGlobals
from ..evaluation import EvaluationContext
from ..evaluation import Package
from ..evaluation import register_bzl_library
from ..label import CMakeTarget
from ..label import Label
from ..label import label_to_generated_cmake_target
from ..label import RelativeLabel
from ..provider import BuildSettingProvider
from ..provider import CMakeDepsProvider
from ..provider import ConditionProvider
from ..provider import FilesProvider
from ..provider import TargetInfo
from ..util import cmake_is_true
from ..util import write_file_if_not_already_equal


class BazelSelectsWrapper:
  """"Defines the `selects` object for `BazelSkylibSelectsLibrary`."""

  def __init__(self, context: EvaluationContext):
    self._context = context

  def config_setting_group(
      self,
      name: str,
      match_all: Optional[List[RelativeLabel]] = None,
      match_any: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    del kwargs
    # Bazel ignores visibility for `config_setting` by default.  See
    # `--incompatible_enforce_config_setting_visibility` and
    # `--incompatible_config_setting_private_default_visibility`.
    context = self._context
    package = context.current_package
    assert package is not None
    get_label = package.get_label
    if match_all is None and match_any is None:
      raise ValueError("must specify match_all or match_any")
    if match_all is not None:
      resolved_match_all = [get_label(condition) for condition in match_all]
    else:
      resolved_match_all = None
    if match_any is not None:
      resolved_match_any = [get_label(condition) for condition in match_any]
    else:
      resolved_match_any = None

    def evaluate() -> bool:
      if resolved_match_all is not None:
        return all(
            context.evaluate_condition(condition)
            for condition in resolved_match_all)

      if resolved_match_any is not None:
        return any(
            context.evaluate_condition(condition)
            for condition in resolved_match_any)

      return False

    label = get_label(name)

    context.add_rule(
        label,
        lambda: context.add_analyzed_target(
            label, TargetInfo(ConditionProvider(evaluate()))),
        analyze_by_default=True)


@register_bzl_library("@bazel_skylib//lib:selects.bzl", build=True)
class BazelSkylibSelectsLibrary(BazelGlobals):

  @property
  def bazel_selects(self):
    return BazelSelectsWrapper(self._context)


@register_bzl_library("@bazel_skylib//rules:expand_template.bzl", build=True)
class BazelSkylibExpandTemplateLibrary(BazelGlobals):

  def bazel_expand_template(self,
                            name: str,
                            out: RelativeLabel,
                            visibility: Optional[List[RelativeLabel]] = None,
                            **kwargs):
    context = self._context
    package = context.current_package
    assert package is not None
    label = package.get_label(name)
    out_target = package.get_label(out)
    context.add_rule(
        label,
        lambda: _expand_template_impl(
            cast(Package, package), label, out_target, **kwargs),
        outs=[out_target],
        analyze_by_default=package.analyze_by_default(visibility))


def _expand_template_impl(
    _package: Package,
    _label: Label,
    _out_target: Label,
    template: Configurable[RelativeLabel],
    substitutions: Configurable[Dict[str, str]],
):
  context = _package.context
  cmake_target_pair = context.generate_cmake_target_pair(
      _label, generate_alias=False)
  out_file = context.get_generated_file_path(_out_target)
  context.add_analyzed_target(
      _out_target,
      TargetInfo(
          CMakeDepsProvider([cmake_target_pair.dep]),
          FilesProvider([out_file])))
  template_target = _package.get_label(
      cast(RelativeLabel, context.evaluate_configurable(template)))
  deps: List[CMakeTarget] = []
  template_paths = context.get_file_paths(template_target, deps)
  assert len(template_paths) == 1
  template_path = template_paths[0]
  script_path = os.path.join(os.path.dirname(__file__), "expand_template.py")
  # Write substitutions to a file because CMake does not handle special
  # characters like "\n" in command lines properly.
  subs_path = os.path.join(_package.repo.cmake_binary_dir,
                           f"{cmake_target_pair.target}.subs.json")
  write_file_if_not_already_equal(
      subs_path,
      json.dumps(context.evaluate_configurable(substitutions)).encode("utf-8"))
  deps.append(template_path)
  deps.append(script_path)
  deps.append(subs_path)
  context.builder.addtext(f"""
add_custom_command(
OUTPUT {cmake_builder.quote_path(out_file)}
COMMAND ${{Python3_EXECUTABLE}} {cmake_builder.quote_path(script_path)}
        {cmake_builder.quote_path(template_path)}
        {cmake_builder.quote_path(subs_path)}
        {cmake_builder.quote_path(out_file)}
DEPENDS {cmake_builder.quote_list(deps)}
VERBATIM
)
add_custom_target({cmake_target_pair.target} DEPENDS {cmake_builder.quote_path(out_file)})
""")
  context.add_analyzed_target(_label,
                              TargetInfo(*cmake_target_pair.as_providers()))


@register_bzl_library("@bazel_skylib//rules:copy_file.bzl", build=True)
class BazelSkylibCopyFileLibrary(BazelGlobals):

  def bazel_copy_file(self,
                      name: str,
                      out: RelativeLabel,
                      src: Configurable[RelativeLabel],
                      visibility: Optional[List[RelativeLabel]] = None,
                      **kwargs):
    del kwargs
    context = self._context
    cmake_command = context.workspace.cmake_vars["CMAKE_COMMAND"]
    native_rules.genrule(
        context,
        name=name,
        outs=[out],
        srcs=[src],
        visibility=visibility,
        cmd=f"{cmake_builder.quote_path(cmake_command)} -E copy $< $@")


@register_bzl_library("@bazel_skylib//rules:write_file.bzl", build=True)
class BazelSkylibWriteFileLibrary(BazelGlobals):

  def bazel_write_file(self,
                       name: str,
                       out: str,
                       visibility: Optional[List[RelativeLabel]] = None,
                       **kwargs):
    context = self._context
    package = context.current_package
    assert package is not None
    label = package.get_label(name)
    out_target = package.get_label(out)
    context.add_rule(
        label,
        lambda: _write_file_impl(
            cast(Package, package), label, out_target, **kwargs),
        outs=[out_target],
        analyze_by_default=package.analyze_by_default(visibility))


def _write_file_impl(_package: Package, _label: Label, _out_target: Label,
                     content: Configurable[List[str]], newline: str, **kwargs):
  del kwargs
  context = _package.context
  out_file = context.get_generated_file_path(_out_target)
  context.add_analyzed_target(_out_target,
                              TargetInfo(FilesProvider([out_file])))
  context.add_analyzed_target(_label, TargetInfo())
  if newline == "unix" or (
      newline == "auto" and
      _package.repo.workspace.cmake_vars["CMAKE_SYSTEM_NAME"] != "Windows"):
    nl = "\n"
  else:
    nl = "\r\n"
  text = nl.join(context.evaluate_configurable(cast(Any, content))) + nl
  write_file_if_not_already_equal(out_file, text.encode("utf-8"))


@register_bzl_library("@bazel_skylib//rules:common_settings.bzl", build=True)
class BazelSkylibCommonSettingsLibrary(BazelGlobals):

  def bazel_bool_flag(self,
                      name: str,
                      visibility: Optional[List[RelativeLabel]] = None,
                      **kwargs):
    context = self._context
    package = context.current_package
    assert package is not None
    label = package.get_label(name)
    context.add_rule(
        label,
        lambda: _bool_flag_impl(cast(Package, package), label, **kwargs),
        analyze_by_default=package.analyze_by_default(visibility))

  def bazel_string_flag(self,
                        name: str,
                        visibility: Optional[List[RelativeLabel]] = None,
                        **kwargs):
    context = self._context
    package = context.current_package
    assert package is not None
    label = package.get_label(name)
    context.add_rule(
        label,
        lambda: _string_flag_impl(cast(Package, package), label, **kwargs),
        analyze_by_default=package.analyze_by_default(visibility))


def _bool_flag_impl(_package: Package, _label: Label,
                    build_setting_default: Configurable[bool], **kwargs):
  del kwargs
  context = _package.context
  cmake_name = label_to_generated_cmake_target(
      _label, _package.repo.cmake_project_name).upper()
  existing_value = context.workspace.cmake_vars.get(cmake_name)
  default_value = context.evaluate_configurable(build_setting_default)
  if existing_value is None:
    value = default_value
  else:
    value = cmake_is_true(existing_value)
  context.builder.addtext(
      f"""option({cmake_name} "" {"ON" if default_value else "OFF"})\n""")
  context.add_analyzed_target(_label, TargetInfo(BuildSettingProvider(value)))


def _string_flag_impl(_package: Package, _label: Label,
                      build_setting_default: Configurable[str], **kwargs):
  del kwargs
  context = _package.context
  cmake_name = label_to_generated_cmake_target(
      _label, _package.repo.cmake_project_name).upper()
  existing_value = _package.repo.workspace.cmake_vars.get(cmake_name)
  default_value = _package.context.evaluate_configurable(build_setting_default)
  if existing_value is None:
    value = default_value
  else:
    value = existing_value
  context.builder.addtext(
      f"""option({cmake_name} "" {cmake_builder.quote_string(value)})\n""")
  context.add_analyzed_target(_label, TargetInfo(BuildSettingProvider(value)))
