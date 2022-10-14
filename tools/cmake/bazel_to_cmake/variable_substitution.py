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

# pylint: disable=relative-beyond-top-level

import os
import re
from typing import Optional, List, Dict, Callable, Match

from . import cmake_builder
from .evaluation import Package
from .label import Label


def apply_location_substitutions(package: Package, cmd: str,
                                 relative_to: str) -> str:
  """Substitues $(location) references in `cmd`.

  https://bazel.build/reference/be/make-variables#predefined_label_variables

  Args:
    package: Package to use for label resolution.
    cmd: Source string.
    relative_to: Working directory.
  Returns:
    Modified string.
  """

  def replace_label(m: Match[str]) -> str:
    key = m.group(1)
    target = package.get_label(m.group(2))
    paths = package.context.get_file_paths(target)

    def _get_relpath(path: str):
      rel_path = os.path.relpath(path, relative_to)
      if os.sep != "/":
        rel_path = rel_path.replace(os.sep, "/")
      return rel_path

    rel_paths = [_get_relpath(path) for path in paths]
    if not key.endswith("s"):
      if len(rel_paths) != 1:
        raise ValueError("Expected single file but received: {rel_paths}")
      return rel_paths[0]
    return " ".join(rel_paths)

  return re.sub(
      r"\$\((location|locations|execpath|execpaths|rootpath|rootpaths)\s+([^)]+)\)",
      replace_label, cmd)


MakeVariableSubstitutions = Dict[str, str]
Toolchain = Callable[[cmake_builder.CMakeBuilder], MakeVariableSubstitutions]

TOOLCHAINS: Dict[Label, Toolchain] = {}


def register_toolchain(target: str) -> Callable[[Toolchain], Toolchain]:
  """Registers a toolchain for use with `apply_make_variable_substitutions."""

  def register(toolchain: Toolchain) -> Toolchain:
    TOOLCHAINS[target] = toolchain
    return toolchain

  return register


def apply_make_variable_substitutions(
    builder: cmake_builder.CMakeBuilder,
    cmd: str,
    substitutions: MakeVariableSubstitutions,
    toolchains: Optional[List[Label]] = None) -> str:
  """Applies Bazel Make variable substitutions.

  Args:
    builder: Builder for emitting CMake module imports.
    cmd: Input string.
    substitutions: Substitutions to apply.
    toolchains: Toolchains defining additional substitutions.
  Returns:
    Substituted string.
  """
  for toolchain in (toolchains or []):
    toolchain_impl = TOOLCHAINS.get(toolchain)
    if toolchain_impl is None:
      raise ValueError(f"Toolchain not defined: {toolchain}")
    substitutions.update(toolchain_impl(builder))

  substitutions["$$"] = "\\$"

  pattern = "|".join(re.escape(key) for key in substitutions)

  def replace_var(m: Match[str]) -> str:
    key = m.group(0)
    replacement = substitutions.get(key)
    if replacement is None:
      raise ValueError(f"Undefined make variable: {key}")
    return replacement

  return re.sub(pattern, replace_var, cmd)
