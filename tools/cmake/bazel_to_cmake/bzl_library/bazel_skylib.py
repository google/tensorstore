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

import io
import itertools
import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast

from .. import native_rules_genrule
from ..cmake_builder import CMakeBuilder
from ..cmake_provider import CMakeAddDependenciesProvider
from ..cmake_provider import default_providers
from ..cmake_target import CMakeTarget
from ..evaluation import EvaluationState
from ..starlark.bazel_target import TargetId
from ..starlark.common_providers import BuildSettingProvider
from ..starlark.common_providers import ConditionProvider
from ..starlark.common_providers import FilesProvider
from ..starlark.invocation_context import InvocationContext
from ..starlark.invocation_context import RelativeLabel
from ..starlark.provider import provider
from ..starlark.provider import TargetInfo
from ..starlark.scope_common import ScopeCommon
from ..starlark.select import Configurable
from ..starlark.select import Select
from ..starlark.toolchain import CMAKE_TOOLCHAIN
from ..util import cmake_is_true
from ..util import cmake_is_windows
from ..util import quote_list
from ..util import quote_path
from ..util import quote_string
from ..util import write_file_if_not_already_equal
from .register import ignore_bzl_library
from .register import register_bzl_library

T = TypeVar("T")


ignore_bzl_library("@bazel_skylib//:bzl_library.bzl")


class BazelSelectsWrapper:
  """Defines the `selects` object for `BazelSkylibSelectsLibrary`."""

  def __init__(self, context: InvocationContext):
    self._context = context

  def with_or_dict(
      self, input_dict: Dict[Union[RelativeLabel, Tuple[RelativeLabel, ...]], T]
  ) -> Dict[RelativeLabel, T]:
    output_dict: Dict[RelativeLabel, T] = {}
    for key, value in input_dict.items():
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

  def with_or(
      self,
      input_dict: Dict[Union[RelativeLabel, Tuple[RelativeLabel, ...]], T],
      no_match_error=None,
  ) -> Select[T]:
    del no_match_error
    conditions = self.with_or_dict(input_dict)
    return Select({
        self._context.resolve_target_or_label(condition): value
        for condition, value in conditions.items()
    })

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
        lambda: _config_settings_group_impl(
            context, target, resolved_match_all, resolved_match_any
        ),
        analyze_by_default=True,
    )


def _config_settings_group_impl(
    _context: InvocationContext,
    _target: TargetId,
    match_all: Optional[List[TargetId]],
    match_any: Optional[List[TargetId]],
):
  def evaluate() -> bool:
    if match_all is not None:
      return all(
          _context.evaluate_condition(condition) for condition in match_all
      )
    if match_any is not None:
      return any(
          _context.evaluate_condition(condition) for condition in match_any
      )
    return False

  _context.add_analyzed_target(
      _target, TargetInfo(ConditionProvider(evaluate()))
  )


@register_bzl_library("@bazel_skylib//lib:selects.bzl")
class BazelSkylibSelectsLibrary(ScopeCommon):

  @property
  def bazel_selects(self):
    return BazelSelectsWrapper(self._context)


@register_bzl_library("@bazel_skylib//rules:expand_template.bzl")
class BazelSkylibExpandTemplateLibrary(ScopeCommon):

  def bazel_expand_template(
      self,
      name: str,
      out: RelativeLabel,
      visibility: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    context = self._context.snapshot()
    target = context.resolve_target(name)
    out_target: TargetId = context.resolve_target_or_label(out)

    context.add_rule(
        target,
        lambda: _expand_template_impl(context, target, out_target, **kwargs),
        outs=[out_target],
        visibility=visibility,
    )


def _expand_template_impl(
    _context: InvocationContext,
    _target: TargetId,
    _out_target: TargetId,
    template: Configurable[RelativeLabel],
    substitutions: Configurable[Dict[str, str]],
):
  state: EvaluationState = _context.access(EvaluationState)

  cmake_target_pair = state.generate_cmake_target_pair(_target).with_alias(None)
  out_file = str(_context.get_generated_file_path(_out_target))

  _context.add_analyzed_target(
      _out_target,
      TargetInfo(
          CMakeAddDependenciesProvider(cmake_target_pair.dep),
          FilesProvider([out_file]),
      ),
  )

  resolved_template = _context.resolve_target_or_label(
      cast(RelativeLabel, _context.evaluate_configurable(template))
  )

  template_collector = state.collect_targets([resolved_template])

  add_dependencies: set[CMakeTarget] = set(
      itertools.chain(
          template_collector.add_dependencies(),
          template_collector.file_paths(),
      )
  )
  template_paths = list(template_collector.file_paths())
  assert len(template_paths) == 1, f"For {template_collector}"

  template_path = template_paths[0]
  script_path = pathlib.PurePath(__file__).parent.joinpath("expand_template.py")

  # Write substitutions to a file because CMake does not handle special
  # characters like "\n" in command lines properly.
  subs_path = state.active_repo.cmake_binary_dir.joinpath(
      f"{cmake_target_pair.target}.subs.json"
  )
  write_file_if_not_already_equal(
      subs_path,
      json.dumps(_context.evaluate_configurable(substitutions)).encode("utf-8"),
  )
  add_dependencies.add(CMakeTarget(template_path))
  add_dependencies.add(CMakeTarget(script_path.as_posix()))
  add_dependencies.add(CMakeTarget(subs_path.as_posix()))

  quoted_output = quote_path(out_file)

  out = io.StringIO()

  out.write(f"""
# expand_template({_target.as_label()})
add_custom_command(
OUTPUT {quoted_output}
COMMAND ${{Python3_EXECUTABLE}} {quote_path(script_path)}
        {quote_path(template_path)}
        {quote_path(subs_path)}
        {quoted_output}
VERBATIM
DEPENDS {quote_list(sorted(add_dependencies))}
COMMENT "Generating {out_file}"
)
set_source_files_properties({quoted_output} PROPERTIES GENERATED TRUE)
add_custom_target({cmake_target_pair.target} DEPENDS {quoted_output})
""")

  _context.access(CMakeBuilder).addtext(out.getvalue())
  _context.add_analyzed_target(
      _target,
      TargetInfo(*default_providers(cmake_target_pair)),
  )


@register_bzl_library("@bazel_skylib//rules:copy_file.bzl")
class BazelSkylibCopyFileLibrary(ScopeCommon):

  def bazel_copy_file(
      self,
      name: str,
      out: RelativeLabel,
      src: Configurable[RelativeLabel],
      visibility: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    del kwargs

    native_rules_genrule.genrule(
        self._context,
        name=name,
        outs=[out],
        srcs=[src],
        visibility=visibility,
        toolchains=[CMAKE_TOOLCHAIN],
        cmd="$(CMAKE_COMMAND) -E copy $< $@",
    )


@register_bzl_library("@bazel_skylib//rules:write_file.bzl")
class BazelSkylibWriteFileLibrary(ScopeCommon):

  def bazel_write_file(
      self,
      name: str,
      out: str,
      visibility: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    context = self._context.snapshot()
    target = context.resolve_target(name)
    out_target: TargetId = context.resolve_target(out)

    context.add_rule(
        target,
        lambda: _write_file_impl(context, target, out_target, **kwargs),
        outs=[out_target],
        visibility=visibility,
    )


def _write_file_impl(
    _context: InvocationContext,
    _target: TargetId,
    _out_target: TargetId,
    content: Optional[Configurable[List[str]]] = None,
    newline: Optional[Configurable[str]] = None,
    **kwargs,
):
  del kwargs
  out_file = str(_context.get_generated_file_path(_out_target))
  _context.add_analyzed_target(
      _out_target, TargetInfo(FilesProvider([out_file]))
  )

  _context.add_analyzed_target(_target, TargetInfo())

  resolved_newline = "auto"
  if newline is not None:
    resolved_newline = _context.evaluate_configurable(newline)

  if resolved_newline == "unix" or (
      resolved_newline == "auto"
      and not cmake_is_windows(
          _context.access(EvaluationState).workspace.cmake_vars[
              "CMAKE_SYSTEM_NAME"
          ]
      )
  ):
    nl = "\n"
  else:
    nl = "\r\n"

  text = ""
  if content is not None:
    text = nl.join(cast(Any, _context.evaluate_configurable_list(content))) + nl

  _context.access(CMakeBuilder).addtext(
      f"\n# bazel_to_cmake wrote {out_file}\n"
  )
  write_file_if_not_already_equal(
      pathlib.PurePath(out_file), text.encode("utf-8")
  )


@register_bzl_library("@bazel_skylib//rules:common_settings.bzl")
class BazelSkylibCommonSettingsLibrary(ScopeCommon):

  BuildSettingInfo = provider(
      doc="A singleton provider that contains the raw value of a build setting",
      fields={
          "value": (
              "The value of the build setting in the current configuration. "
              "This value may come from the command line or an upstream "
              "transition, or else it will be the build setting's default."
          ),
      },
  )

  @property
  def bazel_BuildSettingInfo(self):
    return self.BuildSettingInfo

  def bazel_bool_flag(
      self,
      name: str,
      visibility: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    context = self._context.snapshot()
    target = context.resolve_target(name)
    context.add_rule(
        target,
        lambda: _bool_flag_impl(context, target, **kwargs),
        visibility=visibility,
    )

  def bazel_string_flag(
      self,
      name: str,
      visibility: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    context = self._context.snapshot()
    target = context.resolve_target(name)
    context.add_rule(
        target,
        lambda: _string_flag_impl(context, target, **kwargs),
        visibility=visibility,
    )


def _bool_flag_impl(
    _context: InvocationContext,
    _target: TargetId,
    build_setting_default: Configurable[bool],
    **kwargs,
):
  del kwargs
  active_repo = _context.access(EvaluationState).active_repo

  cmake_name = str(
      active_repo.repository.get_cmake_target_pair(_target).target
  ).upper()
  existing_value = active_repo.workspace.cmake_vars.get(cmake_name)
  default_value = _context.evaluate_configurable(build_setting_default)
  if existing_value is None:
    value = default_value
  else:
    value = cmake_is_true(existing_value)
  _context.access(CMakeBuilder).addtext(
      f"""option({cmake_name} "" {"ON" if default_value else "OFF"})\n"""
  )
  _context.add_analyzed_target(_target, TargetInfo(BuildSettingProvider(value)))


def _string_flag_impl(
    _context: InvocationContext,
    _target: TargetId,
    build_setting_default: Configurable[str],
    **kwargs,
):
  del kwargs
  active_repo = _context.access(EvaluationState).active_repo

  cmake_name = str(
      active_repo.repository.get_cmake_target_pair(_target).target
  ).upper()
  existing_value = active_repo.workspace.cmake_vars.get(cmake_name)
  default_value = _context.evaluate_configurable(build_setting_default)
  if existing_value is None:
    value = default_value
  else:
    value = existing_value
  _context.access(CMakeBuilder).addtext(
      f"""option({cmake_name} "" {quote_string(value)})\n"""
  )
  _context.add_analyzed_target(_target, TargetInfo(BuildSettingProvider(value)))
