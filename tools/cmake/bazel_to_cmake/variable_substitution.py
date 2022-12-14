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
import os
import re
from typing import List, Optional, Match

from .cmake_target import CMakeDepsProvider
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetProvider
from .starlark.bazel_target import TargetId
from .starlark.common_providers import FilesProvider
from .starlark.invocation_context import InvocationContext
from .starlark.toolchain import get_toolchain_substitutions
from .starlark.toolchain import MakeVariableSubstitutions

_LOCATION_RE = re.compile(
    r"^(location|locations|execpath|execpaths|rootpath|rootpaths)\s+(.*)$")

_LOCATION_SUB_RE = re.compile(
    r"\$\((location|locations|execpath|execpaths|rootpath|rootpaths)\s+([^)]+)\)"
)


def _get_location_replacement(_context: InvocationContext, relative_to: str,
                              custom_target_deps: Optional[List[CMakeTarget]],
                              key: str, label: str) -> str:
  """Returns a $(location) replacement for the given key and label."""

  def _get_relpath(path: str):
    rel_path = os.path.relpath(path, relative_to)
    if os.sep != "/":
      rel_path = rel_path.replace(os.sep, "/")
    return rel_path

  target = _context.resolve_target(label)

  info = _context.get_target_info(target)
  cmake_info = info.get(CMakeDepsProvider)
  if custom_target_deps is not None:
    if cmake_info is not None:
      custom_target_deps.extend(cmake_info.targets)

  files_provider = info.get(FilesProvider)
  if files_provider is not None:
    rel_paths = [_get_relpath(path) for path in files_provider.paths]
    if not key.endswith("s"):
      if len(rel_paths) != 1:
        raise ValueError("Expected single file but received: {rel_paths}")
      return rel_paths[0]
    return " ".join(rel_paths)

  cmake_target_provider = info.get(CMakeTargetProvider)
  if cmake_target_provider is not None:
    return f"$<TARGET_FILE:{cmake_target_provider.target}>"

  raise ValueError(
      f"apply_location_substitutions failed for {target} info {repr(info)}")


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

  substitutions = get_toolchain_substitutions(_context, toolchains,
                                              substitutions)

  def _get_replacement(name):
    replacement = substitutions.get(name)
    if replacement is None:
      raise ValueError(
          f"Undefined make variable: '{name}' in {cmd} with {substitutions}")
    return replacement

  # NOTE: location and make variable substitutions do not compose well since
  # for location substitutions to work correctly CMake generator expressions
  # are needed.
  def _do_replacements(cmd):
    out = io.StringIO()
    while True:
      i = cmd.find("$")
      if i == -1:
        out.write(cmd)
        return out.getvalue()
      out.write(cmd[:i])
      j = i + 1
      if cmd[j] == "(":
        # Multi character literal.
        j = cmd.find(")", i + 2)
        assert j > (i + 2)
        name = cmd[i + 2:j]
        m = None
        if enable_location:
          m = _LOCATION_RE.fullmatch(cmd[i + 2:j])
        if m:
          out.write(
              _get_location_replacement(_context, relative_to,
                                        custom_target_deps, m.group(1),
                                        m.group(2)))
        else:
          out.write(_get_replacement(name))
      elif cmd[j] == "$":
        # Escaped $
        out.write("$")
      else:
        # Single letter literal.
        out.write(_get_replacement(cmd[j]))
      cmd = cmd[j + 1:]

  return _do_replacements(cmd)


def apply_make_variable_substitutions(
    _context: InvocationContext,
    cmd: str,
    substitutions: MakeVariableSubstitutions,
    toolchains: Optional[List[TargetId]] = None) -> str:
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
      enable_location=False)


def apply_location_and_make_variable_substitutions(
    _context: InvocationContext, *, cmd: str, relative_to: str,
    custom_target_deps: Optional[List[CMakeTarget]],
    substitutions: MakeVariableSubstitutions,
    toolchains: Optional[List[TargetId]]) -> str:
  """Applies $(location) and Bazel Make variable substitutions."""
  return _apply_location_and_make_variable_substitutions(
      _context,
      cmd=cmd,
      relative_to=relative_to,
      custom_target_deps=custom_target_deps,
      substitutions=substitutions,
      toolchains=toolchains,
      enable_location=True)


def apply_location_substitutions(
    _context: InvocationContext,
    cmd: str,
    relative_to: str,
    custom_target_deps: Optional[List[CMakeTarget]] = None) -> str:
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
    return _get_location_replacement(_context, relative_to, custom_target_deps,
                                     m.group(1), m.group(2))

  return _LOCATION_SUB_RE.sub(_replace, cmd)
