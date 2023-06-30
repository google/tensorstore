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
"""Implements $(location) and Make variable substitution."""

# pylint: disable=relative-beyond-top-level,invalid-name

import io
import json
import os
import pathlib
import re
from typing import Dict, List, Match, Optional

from .cmake_target import CMakeExecutableTargetProvider
from .cmake_target import CMakeLibraryTargetProvider
from .cmake_target import CMakeTarget
from .evaluation import EvaluationState
from .starlark.bazel_target import TargetId
from .starlark.common_providers import FilesProvider
from .starlark.invocation_context import InvocationContext
from .starlark.toolchain import get_toolchain_substitutions
from .starlark.toolchain import MakeVariableSubstitutions

_LOCATION_RE = re.compile(
    r"^(location|locations|execpath|execpaths|rootpath|rootpaths)\s+(.*)$"
)

_LOCATION_SUB_RE = re.compile(
    r"\$\((location|locations|execpath|execpaths|rootpath|rootpaths)\s+([^)]+)\)"
)


def _get_location_replacement(
    _context: InvocationContext,
    _cmd: str,
    relative_to: str,
    custom_target_deps: Optional[List[CMakeTarget]],
    key: str,
    label: str,
) -> str:
  """Returns a $(location) replacement for the given key and label."""

  def _get_relpath(path: str):
    rel_path = os.path.relpath(path, relative_to)
    if os.sep != "/":
      rel_path = rel_path.replace(os.sep, "/")
    return rel_path

  target = _context.resolve_target(label)
  state = _context.access(EvaluationState)

  # First-party references must exist.
  if _context.caller_package_id.repository_id == target.repository_id:
    info = state.get_target_info(target)
  else:
    info = state.get_optional_target_info(target)

  if not info:
    # This target is not available; construct an ephemeral reference.
    cmake_target = state.generate_cmake_target_pair(target)
    if custom_target_deps is not None:
      custom_target_deps.append(cmake_target.dep)
    return f"$<TARGET_FILE:{cmake_target.target}>"

  files_provider = info.get(FilesProvider)
  if files_provider is not None:
    rel_paths = [_get_relpath(path) for path in files_provider.paths]
    if not key.endswith("s"):
      if len(rel_paths) != 1:
        raise ValueError("Expected single file but received: {rel_paths}")
      return rel_paths[0]
    return " ".join(rel_paths)

  cmake_target_provider = info.get(CMakeExecutableTargetProvider)
  if cmake_target_provider is None:
    cmake_target_provider = info.get(CMakeLibraryTargetProvider)
  if cmake_target_provider is not None:
    return f"$<TARGET_FILE:{cmake_target_provider.target}>"

  raise ValueError(
      f'Make location replacement failed: "{json.dumps(_cmd)}" with '
      f"key={key}, target={target.as_label()}, TargetInfo={repr(info)}"
  )


def _apply_location_and_make_variable_substitutions(
    _context: InvocationContext,
    *,
    cmd: str,
    relative_to: str,
    custom_target_deps: Optional[List[CMakeTarget]],
    substitutions: MakeVariableSubstitutions,
    toolchains: Optional[List[TargetId]],
    enable_location: bool,
) -> str:
  """Applies $(location) and Bazel Make variable substitutions."""
  if toolchains is None:
    toolchains = []

  substitutions = get_toolchain_substitutions(
      _context, toolchains, substitutions
  )

  def _get_replacement(name):
    replacement = substitutions.get(name)
    if replacement is None:
      raise ValueError(
          f"Undefined make variable: '{name}' in {json.dumps(cmd)} with"
          f" {substitutions}"
      )
    return replacement

  # NOTE: location and make variable substitutions do not compose well since
  # for location substitutions to work correctly CMake generator expressions
  # are needed.
  def _do_replacements(_cmd):
    out = io.StringIO()
    while True:
      i = _cmd.find("$")
      if i == -1:
        out.write(_cmd)
        return out.getvalue()
      out.write(_cmd[:i])
      j = i + 1
      if _cmd[j] == "(":
        # Multi character literal.
        j = _cmd.find(")", i + 2)
        assert j > (i + 2)
        name = _cmd[i + 2 : j]
        m = None
        if enable_location:
          m = _LOCATION_RE.fullmatch(_cmd[i + 2 : j])
        if m:
          out.write(
              _get_location_replacement(
                  _context,
                  cmd,
                  relative_to,
                  custom_target_deps,
                  m.group(1),
                  m.group(2),
              )
          )
        else:
          out.write(_get_replacement(name))
      elif _cmd[j] == "$":
        # Escaped $
        out.write("$")
      else:
        # Single letter literal.
        out.write(_get_replacement(_cmd[j]))
      _cmd = _cmd[j + 1 :]

  return _do_replacements(cmd)


def apply_make_variable_substitutions(
    _context: InvocationContext,
    cmd: str,
    substitutions: MakeVariableSubstitutions,
    toolchains: Optional[List[TargetId]] = None,
) -> str:
  """Applies Bazel Make variable substitutions.

  Args:
    _context: Context for resolving toolchain substitutions.
    cmd: Input string.
    substitutions: Substitutions to apply.
    toolchains: Toolchains defining additional substitutions.

  Returns:
    Substituted string.
  """
  return _apply_location_and_make_variable_substitutions(
      _context,
      cmd=cmd,
      relative_to="",
      custom_target_deps=None,
      substitutions=substitutions,
      toolchains=toolchains,
      enable_location=False,
  )


def apply_location_and_make_variable_substitutions(
    _context: InvocationContext,
    *,
    cmd: str,
    relative_to: str,
    custom_target_deps: Optional[List[CMakeTarget]],
    substitutions: MakeVariableSubstitutions,
    toolchains: Optional[List[TargetId]],
) -> str:
  """Applies $(location) and Bazel Make variable substitutions."""
  return _apply_location_and_make_variable_substitutions(
      _context,
      cmd=cmd,
      relative_to=relative_to,
      custom_target_deps=custom_target_deps,
      substitutions=substitutions,
      toolchains=toolchains,
      enable_location=True,
  )


def apply_location_substitutions(
    _context: InvocationContext,
    cmd: str,
    relative_to: str,
    custom_target_deps: Optional[List[CMakeTarget]] = None,
) -> str:
  """Substitues $(location) references in `cmd`.

  https://bazel.build/reference/be/make-variables#predefined_label_variables

  Args:
    _context: InvocationContext used for label resolution.
    cmd: Source string.
    relative_to: Working directory.
    custom_target_deps: cmake target dependencies for the genrule

  Returns:
    Modified string.
  """

  def _replace(m: Match[str]) -> str:
    return _get_location_replacement(
        _context, cmd, relative_to, custom_target_deps, m.group(1), m.group(2)
    )

  return _LOCATION_SUB_RE.sub(_replace, cmd)


def generate_substitutions(
    _context: InvocationContext,
    _target: TargetId,
    *,
    src_files: List[str],
    out_files: List[str],
) -> Dict[str, str]:
  # https://bazel.build/reference/be/make-variables
  # https://github.com/bazelbuild/examples/tree/main/make-variables#predefined-path-variables
  #
  # $(BINDIR): bazel-out/x86-fastbuild/bin
  # $(GENDIR): bazel-out/x86-fastbuild/bin
  # $(RULEDIR): bazel-out/x86-fastbuild/bin/testapp
  #
  # Multiple srcs, outs:
  #   $(SRCS): testapp/show_genrule_variables1.src testapp/...
  #   $(OUTS): bazel-out/x86-fastbuild/bin/testapp/subdir/show_genrule_variables1.out bazel-out/...
  #   $(@D): bazel-out/x86-fastbuild/bin/testapp
  #
  # Single srcs, outs
  #   $<: testapp/show_genrule_variables1.src
  #   $@: bazel-out/x86-fastbuild/bin/testapp/subdir/single_file_genrule.out
  #   $(@D): bazel-out/x86-fastbuild/bin/testapp/subdir
  #
  # TODO(jbms): Add missing variables, including:
  #   "$(COMPILATION_MODE)"
  #   "$(TARGET_CPU)"

  source_directory = _context.resolve_source_root(
      _context.caller_package_id.repository_id
  )

  def _relative(path: str) -> pathlib.PurePath:
    nonlocal source_directory
    return pathlib.PurePath(os.path.relpath(path, str(source_directory)))

  binary_dir = pathlib.PurePosixPath(
      _context.resolve_output_root(_target.repository_id)
  )
  package_binary_dir = binary_dir.joinpath(_target.package_name)

  relative_src_files = [
      json.dumps(_relative(path).as_posix()) for path in src_files
  ]
  quoted_out_files = [json.dumps(path) for path in out_files]

  substitutions: Dict[str, str] = {
      "GENDIR": str(binary_dir),
      "BINDIR": str(binary_dir),
      "SRCS": " ".join(relative_src_files),
      "OUTS": " ".join(quoted_out_files),
      "RULEDIR": str(package_binary_dir),
      "@D": str(package_binary_dir),
  }

  if len(src_files) == 1:
    substitutions["<"] = relative_src_files[0]

  if len(out_files) == 1:
    substitutions["@"] = quoted_out_files[0]
    substitutions["@D"] = json.dumps(os.path.dirname(out_files[0]))

  return substitutions
