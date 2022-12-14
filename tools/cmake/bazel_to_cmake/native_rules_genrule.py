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
import re
from typing import List, Optional, cast

from . import cmake_builder
from .cmake_builder import CMakeBuilder
from .cmake_builder import quote_path
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
from .starlark.provider import TargetInfo
from .starlark.select import Configurable
from .variable_substitution import apply_location_and_make_variable_substitutions


@register_native_build_rule
def genrule(self: InvocationContext,
            name: str,
            outs: List[RelativeLabel],
            visibility: Optional[List[RelativeLabel]] = None,
            **kwargs):
  _context = self.snapshot()
  target = _context.resolve_target(name)
  out_targets = _context.resolve_target_or_label_list(outs)

  _context.add_rule(
      target,
      lambda: _genrule_impl(_context, target, out_targets, **kwargs),
      outs=out_targets,
      visibility=visibility)


def _get_relative_path(path: str, relative_to: str) -> str:
  rel_path = os.path.relpath(path, relative_to)
  if os.sep != "/":
    rel_path = rel_path.replace(os.sep, "/")
  return rel_path


def _genrule_impl(_context: InvocationContext,
                  _target: TargetId,
                  _out_targets: List[TargetId],
                  cmd: Configurable[str],
                  srcs: Optional[Configurable[List[RelativeLabel]]] = None,
                  message: Optional[Configurable[str]] = None,
                  toolchains: Optional[List[RelativeLabel]] = None,
                  **kwargs):
  del kwargs
  # Resolve srcs & toolchains
  resolved_srcs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(srcs))
  resolved_toolchains = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(toolchains))

  state = _context.access(EvaluationState)

  cmake_target_pair = state.generate_cmake_target_pair(_target).with_alias(None)

  # Add outputs with a dependency on this target.
  cmake_deps_provider = CMakeDepsProvider([cmake_target_pair.target])
  out_files: List[str] = []
  for out_target in _out_targets:
    out_file = state.get_generated_file_path(out_target)
    out_files.append(out_file)
    _context.add_analyzed_target(
        out_target, TargetInfo(FilesProvider([out_file]), cmake_deps_provider))

  message_text = ""
  if message is not None:
    message_text = _context.evaluate_configurable(message)

  cmake_deps: List[CMakeTarget] = []
  src_files = state.get_targets_file_paths(resolved_srcs, cmake_deps)
  cmake_deps.extend(cast(List[CMakeTarget], src_files))

  source_directory = _context.resolve_source_root(
      _context.caller_package_id.repository_id)
  relative_source_paths = [
      quote_path(_get_relative_path(path, source_directory))
      for path in src_files
  ]
  relative_out_paths = [
      quote_path(_get_relative_path(path, source_directory))
      for path in out_files
  ]

  package_binary_dir = str(
      pathlib.PurePosixPath(
          _context.resolve_output_root(_target.repository_id)).joinpath(
              _target.package_name))

  substitutions = {
      "SRCS":
          " ".join(relative_source_paths),
      "OUTS":
          " ".join(relative_out_paths),
      "RULEDIR":
          _get_relative_path(package_binary_dir, source_directory),
      "GENDIR":
          _get_relative_path(
              _context.resolve_output_root(_target.repository_id),
              source_directory),
  }

  if len(out_files) == 1:
    substitutions["@"] = relative_out_paths[0]
  if len(src_files) == 1:
    substitutions["<"] = relative_source_paths[0]

  # https://bazel.build/reference/be/make-variables
  # TODO(jbms): Add missing variables, including:
  #   "$(BINDIR)"
  #   "$(COMPILATION_MODE)"
  #   "$(TARGET_CPU)"
  #   "$(@D)"
  cmd_text = apply_location_and_make_variable_substitutions(
      _context,
      cmd=_context.evaluate_configurable(cmd),
      relative_to=source_directory,
      custom_target_deps=cmake_deps,
      substitutions=substitutions,
      toolchains=resolved_toolchains)

  builder = _context.access(CMakeBuilder)
  builder.addtext(f"\n# {_target.as_label()}")
  _emit_genrule(
      builder=builder,
      cmake_target=cmake_deps_provider.targets[0],
      out_files=out_files,
      cmake_deps=cmake_deps,
      cmd_text=cmd_text,
      message=message_text)
  _context.add_analyzed_target(_target, TargetInfo(cmake_deps_provider))


_QUOTED_VAR = re.compile(r"(^[$]{[A-Z0-9_]+})|(^\"[$]{[A-Z0-9_]+}\")")


def _emit_genrule(
    builder: CMakeBuilder,
    cmake_target: str,
    out_files: List[str],
    cmake_deps: List[CMakeTarget],
    cmd_text: str,
    message: Optional[str] = None,
):
  cmd_text = cmd_text.strip()
  if message:
    optional_message_text = f"COMMENT {cmake_builder.quote_string(message)}\n  "
  else:
    optional_message_text = ""

  # In bazel, genrules run under bash shell, however in CMake it's more
  # complicated. Assume that if the command starts with a quoted CMake
  # variable that it's not necessary.
  if not _QUOTED_VAR.match(cmd_text):
    bash = os.getenv("BAZEL_SH")
    if bash is not None:
      cmd_text = f"$ENV{{BAZEL_SH}} -c {quote_string(cmd_text)}"
    else:
      cmd_text = f"bash -c {quote_string(cmd_text)}"

  builder.addtext(f"""
add_custom_command(
  OUTPUT {cmake_builder.quote_list(out_files)}
  DEPENDS {cmake_builder.quote_list(cast(List[str], cmake_deps))}
  COMMAND {cmd_text}
  {optional_message_text}VERBATIM
  WORKING_DIRECTORY "${{CMAKE_CURRENT_SOURCE_DIR}}"
)
add_custom_target({cmake_target} DEPENDS {cmake_builder.quote_list(out_files)})
""")
