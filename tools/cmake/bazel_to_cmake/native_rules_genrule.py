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
import re
from typing import List, Optional, cast

from .cmake_builder import CMakeBuilder
from .cmake_builder import quote_list
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
from .variable_substitution import generate_substitutions


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

  # Add outputs with a dependency on this target.
  cmake_deps_provider = CMakeDepsProvider([cmake_target_pair.target])
  out_files: List[str] = []
  for out_target in _out_targets:
    out_file = str(state.get_generated_file_path(out_target))
    out_files.append(out_file)
    _context.add_analyzed_target(
        out_target, TargetInfo(FilesProvider([out_file]), cmake_deps_provider)
    )

  cmake_deps: List[CMakeTarget] = []
  cmake_deps.extend(state.get_deps(resolved_tools))

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

  builder = _context.access(CMakeBuilder)
  builder.addtext(f"\n# genrule({_target.as_label()})")
  _emit_genrule(
      builder=builder,
      cmake_target=cmake_deps_provider.targets[0],
      out_files=out_files,
      cmake_deps=cmake_deps,
      cmd_text=cmd_text,
      message=message_text,
  )
  _context.add_analyzed_target(_target, TargetInfo(cmake_deps_provider))


_QUOTED_VAR = re.compile(r"(^[$]{[A-Z0-9_]+})|(^\"[$]{[A-Z0-9_]+}\")")


def _emit_genrule(
    builder: CMakeBuilder,
    cmake_target: str,
    out_files: List[str],
    cmake_deps: List[CMakeTarget],
    cmd_text: str,
    message: Optional[str],
):
  cmd_text = cmd_text.strip()
  if message:
    optional_message_text = f"COMMENT {quote_string(message)}\n  "
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

  sep = "\n    "
  builder.addtext(f"""
add_custom_command(
  OUTPUT{sep}{quote_list(out_files, sep)}
  DEPENDS{sep}{quote_list(cast(List[str], cmake_deps), sep)}
  COMMAND {cmd_text}
  {optional_message_text}VERBATIM
  WORKING_DIRECTORY "${{CMAKE_CURRENT_SOURCE_DIR}}"
)
add_custom_target({cmake_target} DEPENDS {quote_list(out_files)})
""")
