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
from typing import Any, Dict, List, Optional, cast

from .. import native_rules_genrule
from ..cmake_builder import CMakeBuilder
from ..cmake_builder import quote_list
from ..cmake_builder import quote_path
from ..cmake_builder import quote_string
from ..cmake_target import CMakeDepsProvider
from ..cmake_target import CMakeTarget
from ..cmake_target import label_to_generated_cmake_target
from ..evaluation import EvaluationState
from ..starlark.bazel_globals import BazelGlobals
from ..starlark.bazel_globals import register_bzl_library
from ..starlark.bazel_target import TargetId
from ..starlark.common_providers import BuildSettingProvider
from ..starlark.common_providers import ConditionProvider
from ..starlark.common_providers import FilesProvider
from ..starlark.invocation_context import InvocationContext
from ..starlark.invocation_context import RelativeLabel
from ..starlark.provider import TargetInfo
from ..starlark.select import Configurable
from ..starlark.select import Select
from ..starlark.toolchain import CMAKE_TOOLCHAIN
from ..util import cmake_is_true
from ..util import cmake_is_windows
from ..util import write_file_if_not_already_equal


class BazelSelectsWrapper:
  """Defines the `selects` object for `BazelSkylibSelectsLibrary`."""

  def __init__(self, context: InvocationContext):
    self._context = context

  def with_or_dict(self, input_dict):
    output_dict = {}
    for (key, value) in input_dict.items():
      if isinstance(key, tuple):
        for config_setting in key:
          if config_setting in output_dict.keys():
            raise ValueError(f"key {config_setting} appears multiple times")
          output_dict[config_setting] = value
      else:
        if key in output_dict.keys():
          raise ValueError(f"key {key} appears multiple times")
        output_dict[key] = value
    return output_dict

  def with_or(self, input_dict, no_match_error=None):
    del no_match_error
    return Select(self.with_or_dict(input_dict))

  def config_setting_group(
      self,
      name: str,
      match_all: Optional[List[RelativeLabel]] = None,
      match_any: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    del kwargs
    context = self._context.snapshot()
    target = context.resolve_target(name)

    # Bazel ignores visibility for `config_setting` by default.  See
    # `--incompatible_enforce_config_setting_visibility` and
    # `--incompatible_config_setting_private_default_visibility`.
    if match_all is None and match_any is None:
      raise ValueError("must specify match_all or match_any")

    resolved_match_all = None
    if match_all is not None:
      resolved_match_all = context.resolve_target_or_label_list(match_all)
    resolved_match_any = None
    if match_any is not None:
      resolved_match_any = context.resolve_target_or_label_list(match_any)

    context.add_rule(
        target,
        lambda: _config_settings_group_impl(context, target, resolved_match_all,
                                            resolved_match_any),
        analyze_by_default=True)


def _config_settings_group_impl(_context: InvocationContext, _target: TargetId,
                                match_all: Optional[List[TargetId]],
                                match_any: Optional[List[TargetId]]):

  def evaluate() -> bool:
    if match_all is not None:
      return all(
          _context.evaluate_condition(condition) for condition in match_all)
    if match_any is not None:
      return any(
          _context.evaluate_condition(condition) for condition in match_any)
    return False

  _context.add_analyzed_target(_target,
                               TargetInfo(ConditionProvider(evaluate())))


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
    context = self._context.snapshot()
    target = context.resolve_target(name)
    out_target: TargetId = context.resolve_target_or_label(out)

    context.add_rule(
        target,
        lambda: _expand_template_impl(context, target, out_target, **kwargs),
        outs=[out_target],
        visibility=visibility)


def _expand_template_impl(
    _context: InvocationContext,
    _target: TargetId,
    _out_target: TargetId,
    template: Configurable[RelativeLabel],
    substitutions: Configurable[Dict[str, str]],
):
  state: EvaluationState = _context.access(EvaluationState)

  cmake_target_pair = state.generate_cmake_target_pair(_target).with_alias(None)
  out_file = _context.get_generated_file_path(_out_target)

  _context.add_analyzed_target(
      _out_target,
      TargetInfo(
          CMakeDepsProvider([cmake_target_pair.dep]),
          FilesProvider([out_file])))

  resolved_template = _context.resolve_target_or_label(
      cast(RelativeLabel, _context.evaluate_configurable(template)))

  deps: List[CMakeTarget] = []
  template_paths = state.get_file_paths(resolved_template, deps)

  assert len(template_paths) == 1
  template_path = template_paths[0]
  script_path = os.path.join(os.path.dirname(__file__), "expand_template.py")
  # Write substitutions to a file because CMake does not handle special
  # characters like "\n" in command lines properly.
  subs_path = os.path.join(state.repo.cmake_binary_dir,
                           f"{cmake_target_pair.target}.subs.json")
  write_file_if_not_already_equal(
      subs_path,
      json.dumps(_context.evaluate_configurable(substitutions)).encode("utf-8"))
  deps.append(CMakeTarget(template_path))
  deps.append(CMakeTarget(script_path))
  deps.append(CMakeTarget(subs_path))

  builder: CMakeBuilder = _context.access(CMakeBuilder)
  builder.addtext(f"""
add_custom_command(
OUTPUT {quote_path(out_file)}
COMMAND ${{Python3_EXECUTABLE}} {quote_path(script_path)}
        {quote_path(template_path)}
        {quote_path(subs_path)}
        {quote_path(out_file)}
DEPENDS {quote_list(deps)}
VERBATIM
)
add_custom_target({cmake_target_pair.target} DEPENDS {quote_path(out_file)})
""")
  _context.add_analyzed_target(_target,
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

    native_rules_genrule.genrule(
        self._context,
        name=name,
        outs=[out],
        srcs=[src],
        visibility=visibility,
        toolchains=[CMAKE_TOOLCHAIN],
        cmd="$(CMAKE_COMMAND) -E copy $< $@")


@register_bzl_library("@bazel_skylib//rules:write_file.bzl", build=True)
class BazelSkylibWriteFileLibrary(BazelGlobals):

  def bazel_write_file(self,
                       name: str,
                       out: str,
                       visibility: Optional[List[RelativeLabel]] = None,
                       **kwargs):
    context = self._context.snapshot()
    target = context.resolve_target(name)
    out_target: TargetId = context.resolve_target(out)

    context.add_rule(
        target,
        lambda: _write_file_impl(context, target, out_target, **kwargs),
        outs=[out_target],
        visibility=visibility)


def _write_file_impl(
    _context: InvocationContext,
    _target: TargetId,
    _out_target: TargetId,
    content: Configurable[List[str]],
    newline: Configurable[str],
    **kwargs,
):
  del kwargs
  out_file = _context.get_generated_file_path(_out_target)
  _context.add_analyzed_target(_out_target,
                               TargetInfo(FilesProvider([out_file])))

  _context.add_analyzed_target(_target, TargetInfo())

  resolved_newline = _context.evaluate_configurable(newline)

  if resolved_newline == "unix" or (
      resolved_newline == "auto" and not cmake_is_windows(
          _context.access(
              EvaluationState).workspace.cmake_vars["CMAKE_SYSTEM_NAME"])):
    nl = "\n"
  else:
    nl = "\r\n"
  text = nl.join(cast(Any, _context.evaluate_configurable_list(content))) + nl

  _context.access(CMakeBuilder).addtext(
      f"\n# bazel_to_cmake wrote {out_file}\n")
  write_file_if_not_already_equal(out_file, text.encode("utf-8"))


@register_bzl_library("@bazel_skylib//rules:common_settings.bzl", build=True)
class BazelSkylibCommonSettingsLibrary(BazelGlobals):

  def bazel_bool_flag(self,
                      name: str,
                      visibility: Optional[List[RelativeLabel]] = None,
                      **kwargs):
    context = self._context.snapshot()
    target = context.resolve_target(name)
    context.add_rule(
        target,
        lambda: _bool_flag_impl(context, target, **kwargs),
        visibility=visibility)

  def bazel_string_flag(self,
                        name: str,
                        visibility: Optional[List[RelativeLabel]] = None,
                        **kwargs):
    context = self._context.snapshot()
    target = context.resolve_target(name)
    context.add_rule(
        target,
        lambda: _string_flag_impl(context, target, **kwargs),
        visibility=visibility)


def _bool_flag_impl(_context: InvocationContext, _target: TargetId,
                    build_setting_default: Configurable[bool], **kwargs):
  del kwargs
  repo = _context.access(EvaluationState).repo

  cmake_name = str(
      label_to_generated_cmake_target(_target,
                                      repo._cmake_project_name).target).upper()
  existing_value = repo.workspace.cmake_vars.get(cmake_name)
  default_value = _context.evaluate_configurable(build_setting_default)
  if existing_value is None:
    value = default_value
  else:
    value = cmake_is_true(existing_value)
  _context.access(CMakeBuilder).addtext(
      f"""option({cmake_name} "" {"ON" if default_value else "OFF"})\n""")
  _context.add_analyzed_target(_target, TargetInfo(BuildSettingProvider(value)))


def _string_flag_impl(_context: InvocationContext, _target: TargetId,
                      build_setting_default: Configurable[str], **kwargs):
  del kwargs
  repo = _context.access(EvaluationState).repo

  cmake_name = str(
      label_to_generated_cmake_target(_target,
                                      repo._cmake_project_name).target).upper()
  existing_value = repo.workspace.cmake_vars.get(cmake_name)
  default_value = _context.evaluate_configurable(build_setting_default)
  if existing_value is None:
    value = default_value
  else:
    value = existing_value
  _context.access(CMakeBuilder).addtext(
      f"""option({cmake_name} "" {quote_string(value)})\n""")
  _context.add_analyzed_target(_target, TargetInfo(BuildSettingProvider(value)))
