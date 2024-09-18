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
import itertools
import pathlib
from typing import List, Optional

from .cmake_builder import CMakeBuilder
from .cmake_provider import CMakeAddDependenciesProvider
from .cmake_provider import CMakePackageDepsProvider
from .cmake_provider import make_providers
from .cmake_target import CMakeTarget
from .emit_filegroup import emit_filegroup
from .emit_filegroup import emit_genrule
from .evaluation import EvaluationState
from .starlark import rule  # pylint: disable=unused-import
from .starlark.bazel_build_file import register_native_build_rule
from .starlark.bazel_target import TargetId
from .starlark.common_providers import FilesProvider
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.provider import TargetInfo
from .starlark.select import Configurable
from .util import quote_path_list
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

  srcs_collector = state.collect_targets(resolved_srcs)
  srcs_files = sorted(set(srcs_collector.file_paths()))

  _context.add_analyzed_target(
      _target,
      TargetInfo(
          *make_providers(
              cmake_target_pair,
              CMakePackageDepsProvider,
              CMakeAddDependenciesProvider,
          ),
          FilesProvider(srcs_files),
      ),
  )

  # Also add an INTERFACE_LIBRARY in order to reference in compile targets.
  out = io.StringIO()
  out.write(f"\n# filegroup({_target.as_label()})\n")
  emit_filegroup(
      out,
      cmake_name=cmake_target_pair.target,
      filegroup_files=srcs_files,
      source_directory=repo.source_directory,
      cmake_binary_dir=repo.cmake_binary_dir,
      add_dependencies=set(srcs_collector.add_dependencies()),
  )
  _context.access(CMakeBuilder).addtext(out.getvalue())


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
  cmake_deps_provider = CMakeAddDependenciesProvider(cmake_target_pair.target)
  repo = state.workspace.all_repositories.get(_target.repository_id)
  assert repo is not None

  # Add outputs with a dependency on this target.
  out_files: List[str] = []
  out_dirs: set[pathlib.PurePath] = set()
  for out_target in _out_targets:
    path = state.get_generated_file_path(out_target)
    if path.parent and path.parent != state.active_repo.cmake_binary_dir:
      out_dirs.add(path.parent)
    str_path = str(path)
    out_files.append(str_path)
    _context.add_analyzed_target(
        out_target, TargetInfo(FilesProvider([str_path]), cmake_deps_provider)
    )

  tools_collector = state.collect_deps(resolved_tools)
  src_collector = state.collect_targets(resolved_srcs)

  add_dependencies: List[CMakeTarget] = list(
      itertools.chain(
          tools_collector.targets(),
          src_collector.add_dependencies(),
      )
  )
  add_dependencies.extend(CMakeTarget(x) for x in src_collector.file_paths())

  substitutions = generate_substitutions(
      _context,
      _target,
      src_files=list(src_collector.file_paths()),
      out_files=out_files,
  )
  source_directory = _context.resolve_source_root(
      _context.caller_package_id.repository_id
  )

  cmd_text = apply_location_and_make_variable_substitutions(
      _context,
      cmd=_context.evaluate_configurable(cmd),
      relative_to=str(source_directory),
      add_dependencies=add_dependencies,
      substitutions=substitutions,
      toolchains=resolved_toolchains,
  )

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

  _context.add_analyzed_target(_target, TargetInfo(cmake_deps_provider))

  # NOTE: We could do create the directories here:
  # pathlib.Path(x).mkdir(parents=True, exist_ok=True) for x in out_dirs
  out = io.StringIO()

  out.write(f"\n# genrule({_target.as_label()})\n")
  if out_dirs:
    out.write(f"file(MAKE_DIRECTORY {quote_path_list(sorted(out_dirs))})\n")

  emit_genrule(
      out,
      "genrule__" + cmake_target_pair.target,
      generated_files=out_files,
      add_dependencies=add_dependencies,
      cmd_text=cmd_text,
      message=message_text,
  )
  emit_filegroup(
      out,
      cmake_name=cmake_target_pair.target,
      filegroup_files=out_files,
      source_directory=repo.source_directory,
      cmake_binary_dir=repo.cmake_binary_dir,
      add_dependencies=[CMakeTarget("genrule__" + cmake_target_pair.target)],
  )
  _context.access(CMakeBuilder).addtext(out.getvalue())
