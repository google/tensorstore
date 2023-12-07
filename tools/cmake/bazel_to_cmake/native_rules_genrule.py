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

import io
import pathlib
import re
from typing import List, Optional, cast, Set

from .cmake_builder import CMakeBuilder
from .cmake_builder import quote_list
from .cmake_builder import quote_path_list
from .cmake_builder import quote_string
from .cmake_target import CMakeDepsProvider
from .cmake_target import CMakeTarget
from .evaluation import EvaluationState
from .starlark import rule  # pylint: disable=unused-import
from .starlark.bazel_globals import register_native_build_rule
from .starlark.bazel_target import TargetId
from .starlark.common_providers import FilesProvider
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.provider import Provider
from .starlark.provider import TargetInfo
from .starlark.select import Configurable
from .util import is_relative_to
from .variable_substitution import apply_location_and_make_variable_substitutions
from .variable_substitution import do_bash_command_replacement
from .variable_substitution import generate_substitutions


@register_native_build_rule
def filegroup(
    self: InvocationContext,
    name: str,
    srcs: Optional[List[RelativeLabel]] = None,
    visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  # https://bazel.build/reference/be/general#filegroup
  del kwargs
  # NOTE: Build breaks when filegroup add_rule() uses visibility.
  del visibility
  context = self.snapshot()
  target = context.resolve_target(name)

  context.add_rule(
      target,
      lambda: _filegroup_impl(context, target, srcs=srcs),
      analyze_by_default=False,
  )


def _filegroup_impl(
    _context: InvocationContext,
    _target: TargetId,
    srcs: Optional[List[RelativeLabel]] = None,
):
  resolved_srcs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(srcs)
  )

  state = _context.access(EvaluationState)

  repo = state.workspace.all_repositories.get(_target.repository_id)
  assert repo is not None

  cmake_target_pair = state.generate_cmake_target_pair(_target, alias=False)

  cmake_deps: List[CMakeTarget] = []
  srcs_files = state.get_targets_file_paths(resolved_srcs, cmake_deps)

  # Also add an INTERFACE_LIBRARY in order to reference in compile targets.
  out = io.StringIO()
  out.write(f"\n# filegroup({_target.as_label()})\n")
  _emit_filegroup(
      out,
      cmake_target_pair.target,
      srcs_files,
      repo.source_directory,
      repo.cmake_binary_dir,
      add_dependencies=cmake_deps,
  )
  _context.access(CMakeBuilder).addtext(out.getvalue())

  providers: List[Provider] = [FilesProvider(srcs_files)]
  _context.add_analyzed_target(
      _target, TargetInfo(*cmake_target_pair.as_providers(), *providers)
  )


@register_native_build_rule
def genrule(
    self: InvocationContext,
    name: str,
    outs: List[RelativeLabel],
    visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  _context = self.snapshot()
  target = _context.parse_rule_target(name)
  out_targets = _context.resolve_target_or_label_list(outs)

  _context.add_rule(
      target,
      lambda: _genrule_impl(_context, target, out_targets, **kwargs),
      outs=out_targets,
      visibility=visibility,
  )


def _genrule_impl(
    _context: InvocationContext,
    _target: TargetId,
    _out_targets: List[TargetId],
    cmd: Configurable[str],
    srcs: Optional[Configurable[List[RelativeLabel]]] = None,
    message: Optional[Configurable[str]] = None,
    tools: Optional[List[RelativeLabel]] = None,
    toolchains: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  for x in ["cmd_bash", "cmd_bat", "cmd_ps", "exec_tools"]:
    if x in kwargs:
      raise ValueError(f"genrule cannot handle {kwargs}")

  # In Bazel, `genrule` always invokes bash, even on Windows, unless one of
  # `cmd_{bat,ps}` is specified.  However, as we wish to avoid the Windows CMake
  # build depending on Bash, instead just let CMake use the default shell.

  del kwargs
  state = _context.access(EvaluationState)

  # Resolve srcs & toolchains
  resolved_srcs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(srcs)
  )
  resolved_tools = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(tools)
  )
  resolved_toolchains = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(toolchains)
  )

  message_text = ""
  if message is not None:
    message_text = _context.evaluate_configurable(message)

  cmake_target_pair = state.generate_cmake_target_pair(_target).with_alias(None)
  cmake_deps_provider = CMakeDepsProvider([cmake_target_pair.target])
  repo = state.workspace.all_repositories.get(_target.repository_id)
  assert repo is not None

  # Add outputs with a dependency on this target.
  out_files: List[str] = []
  for out_target in _out_targets:
    path = str(state.get_generated_file_path(out_target))
    out_files.append(path)
    _context.add_analyzed_target(
        out_target, TargetInfo(FilesProvider([path]), cmake_deps_provider)
    )

  cmake_deps: List[CMakeTarget] = state.get_deps(resolved_tools)
  src_files = state.get_targets_file_paths(resolved_srcs, cmake_deps)
  cmake_deps.extend(cast(List[CMakeTarget], src_files))

  substitutions = generate_substitutions(
      _context,
      _target,
      src_files=src_files,
      out_files=out_files,
  )
  source_directory = _context.resolve_source_root(
      _context.caller_package_id.repository_id
  )

  cmd_text = apply_location_and_make_variable_substitutions(
      _context,
      cmd=_context.evaluate_configurable(cmd),
      relative_to=str(source_directory),
      custom_target_deps=cmake_deps,
      substitutions=substitutions,
      toolchains=resolved_toolchains,
  )

  out = io.StringIO()
  out.write(f"\n# genrule({_target.as_label()})\n")

  if cmd_text.find("$(") != -1:
    cmd_text = do_bash_command_replacement(cmd_text)

  if cmd_text.find("$(") != -1:
    # This custom command still has shell processing. Try wrapping it in bash.
    escaped = cmd_text.translate(
        str.maketrans({
            '"': r"\"",
            "\\": r"\\",
            "\n": " ",
        })
    )
    cmd_text = f'bash -c "{escaped}"'

  _emit_genrule(
      out,
      "genrule__" + cmake_target_pair.target,
      generated_files=out_files,
      add_dependencies=cmake_deps,
      cmd_text=cmd_text,
      message=message_text,
  )
  _emit_filegroup(
      out,
      cmake_target_pair.target,
      out_files,
      repo.source_directory,
      repo.cmake_binary_dir,
      add_dependencies=[CMakeTarget("genrule__" + cmake_target_pair.target)],
  )
  _context.access(CMakeBuilder).addtext(out.getvalue())

  _context.add_analyzed_target(_target, TargetInfo(cmake_deps_provider))


_QUOTED_VAR = re.compile(r"(^[$]{[A-Z0-9_]+})|(^\"[$]{[A-Z0-9_]+}\")")


def _emit_filegroup(
    out: io.StringIO,
    cmake_name: str,
    filegroup_files: List[str],
    source_directory: pathlib.PurePath,
    cmake_binary_dir: pathlib.PurePath,
    add_dependencies: Optional[List[CMakeTarget]] = None,
):
  has_proto = False
  has_ch = False
  includes: Set[str] = set()
  for path in filegroup_files:
    has_proto = has_proto or path.endswith(".proto")
    has_ch = (
        has_ch
        or path.endswith(".c")
        or path.endswith(".h")
        or path.endswith(".hpp")
        or path.endswith(".cc")
        or path.endswith(".inc")
    )
    if is_relative_to(pathlib.PurePath(path), source_directory):
      includes.add("${PROJECT_SOURCE_DIR}")
    if is_relative_to(pathlib.PurePath(path), cmake_binary_dir):
      includes.add("${PROJECT_BINARY_DIR}")

  sep = "\n    "
  quoted_includes = quote_list(sorted(includes), sep)
  quoted_srcs = quote_path_list(sorted(filegroup_files), sep)

  out.write(f"add_library({cmake_name} INTERFACE)\n")
  out.write(f"target_sources({cmake_name} INTERFACE{sep}{quoted_srcs})\n")
  if has_proto:
    out.write(
        f"set_property(TARGET {cmake_name} PROPERTY"
        f" INTERFACE_IMPORTS{sep}{quoted_includes})\n"
    )
  if has_ch:
    out.write(
        f"set_property(TARGET {cmake_name} PROPERTY"
        f" INTERFACE_INCLUDE_DIRECTORIES{sep}{quoted_includes})\n"
    )
  if add_dependencies:
    deps_str = sep.join(sorted(set(add_dependencies)))
    out.write(f"add_dependencies({cmake_name} {deps_str})\n")


def _emit_genrule(
    out: io.StringIO,
    cmake_name: str,
    generated_files: List[str],
    add_dependencies: List[CMakeTarget],
    cmd_text: str,
    message: Optional[str],
):
  cmd_text = cmd_text.strip()
  if message:
    optional_message_text = f"COMMENT {quote_string(message)}\n  "
  else:
    optional_message_text = ""

  sep = "\n    "
  quoted_outputs = quote_list(generated_files, sep)
  deps_str = quote_list(sorted(set(add_dependencies)), sep)

  out.write(f"""add_custom_command(
  OUTPUT{sep}{quoted_outputs}
  DEPENDS{sep}{deps_str}
  COMMAND {cmd_text}
  {optional_message_text}VERBATIM
  WORKING_DIRECTORY "${{CMAKE_CURRENT_SOURCE_DIR}}"
)
add_custom_target({cmake_name} DEPENDS{sep}{quoted_outputs})
""")
