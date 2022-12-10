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
"""Implements register_toolchain helers."""

# pylint: disable=relative-beyond-top-level,invalid-name

from typing import Callable, Dict, List

from .bazel_target import parse_absolute_target
from .bazel_target import TargetId
from .invocation_context import InvocationContext

MakeVariableSubstitutions = Dict[str, str]
Toolchain = Callable[[InvocationContext], MakeVariableSubstitutions]

# The CMAKE_TOOLCHAIN is special; it just forwards CMAKE variables.
# Add additional variables as necessary
CMAKE_TOOLCHAIN = "@bazel_to_cmake//:cmake_toolchain"

_CMAKE_DICT = {
    "CMAKE_COMMAND": "${CMAKE_COMMAND}",
}

_TOOLCHAINS: Dict[TargetId, Toolchain] = {
    parse_absolute_target(CMAKE_TOOLCHAIN): lambda x: _CMAKE_DICT,
}


def register_toolchain(target: str) -> Callable[[Toolchain], Toolchain]:
  """Registers a toolchain for use with `apply_make_variable_substitutions."""

  target_id = parse_absolute_target(target)

  def register(toolchain: Toolchain) -> Toolchain:
    assert toolchain is not None
    _TOOLCHAINS[target_id] = toolchain
    return toolchain

  return register


def get_toolchain_substitutions(
    context: InvocationContext, toolchains: List[TargetId],
    substitutions: MakeVariableSubstitutions) -> MakeVariableSubstitutions:
  for toolchain in toolchains:
    toolchain_impl = _TOOLCHAINS.get(toolchain)
    if toolchain_impl is None:
      raise ValueError(f"Toolchain not defined: {repr(toolchain)}")
    substitutions.update(toolchain_impl(context))
  return substitutions
