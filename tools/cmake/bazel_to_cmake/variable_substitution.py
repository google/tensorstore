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

import os
import re
from typing import Callable, Dict, List, Match, Optional

from . import cmake_builder
from .cmake_target import CMakeDepsProvider
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetProvider
from .evaluation import EvaluationState
from .starlark.bazel_target import parse_absolute_target
from .starlark.bazel_target import TargetId
from .starlark.common_providers import FilesProvider
from .starlark.invocation_context import InvocationContext


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

  def _get_relpath(path: str):
    rel_path = os.path.relpath(path, relative_to)
    if os.sep != "/":
      rel_path = rel_path.replace(os.sep, "/")
    return rel_path

  state = _context.access(EvaluationState)

  def replace_label(m: Match[str]) -> str:
    key = m.group(1)
    target = _context.resolve_target(m.group(2))

    info = state.get_target_info(target)
    if custom_target_deps is not None:
      cmake_info = info.get(CMakeDepsProvider)
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
      return cmake_target_provider.target
    raise ValueError(
        f"apply_location_substitutions failed for {target} info {repr(info)}")

  return re.sub(
      r"\$\((location|locations|execpath|execpaths|rootpath|rootpaths)\s+([^)]+)\)",
      replace_label, cmd)


MakeVariableSubstitutions = Dict[str, str]
Toolchain = Callable[[cmake_builder.CMakeBuilder], MakeVariableSubstitutions]

TOOLCHAINS: Dict[TargetId, Toolchain] = {}


def register_toolchain(target: str) -> Callable[[Toolchain], Toolchain]:
  """Registers a toolchain for use with `apply_make_variable_substitutions."""

  target_id = parse_absolute_target(target)

  def register(toolchain: Toolchain) -> Toolchain:
    assert toolchain is not None
    TOOLCHAINS[target_id] = toolchain
    return toolchain

  return register


def apply_make_variable_substitutions(
    builder: cmake_builder.CMakeBuilder,
    cmd: str,
    substitutions: MakeVariableSubstitutions,
    toolchains: Optional[List[TargetId]] = None) -> str:
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
      raise ValueError(f"Toolchain not defined: {repr(toolchain)}")
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
