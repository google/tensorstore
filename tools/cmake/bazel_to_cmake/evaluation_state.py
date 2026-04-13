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
"""Defines the EvaluationState interface to break circular dependencies."""

from collections.abc import Iterable
import pathlib
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable

from .active_repository import Repository
from .cmake_target import CMakeTargetPair
from .provider_util import ProviderCollection
from .starlark.bazel_target import TargetId
from .starlark.provider import TargetInfo
from .workspace import Workspace

T = TypeVar("T")


@runtime_checkable
class EvaluationState(Protocol):
  """Interface for EvaluationImpl used by bzl_library rules."""

  workspace: Workspace
  active_repo: Repository

  def collect_targets(
      self,
      targets: Iterable[TargetId],
      collector: ProviderCollection | None = None,
  ) -> ProviderCollection:
    ...

  def get_optional_target_info(self, target_id: TargetId) -> TargetInfo | None:
    ...

  def get_target_info(self, target_id: TargetId) -> TargetInfo:
    ...

  def generate_cmake_target_pair(
      self, target_id: TargetId, alias: bool = True
  ) -> CMakeTargetPair:
    ...

  def get_placeholder_source(self) -> str:
    ...

  def call_after_analysis(self, callback: Callable[[], None]) -> None:
    ...

  @property
  def required_dep_packages(self) -> Iterable[str]:
    ...

  @property
  def evaluation_context(self) -> Any:
    ...


def get_fetch_content_base_dir(state: EvaluationState) -> pathlib.PurePath:
  # The details here depend a bit on how we configure fetch-content.
  # https://github.com/Kitware/CMake/blob/master/Modules/FetchContent.cmake
  #
  # Generally we write a "proxy" cmake file that is patched into the
  # SOURCE_DIR, which then generates another "build_rules.cmake".
  #
  # FetchContent, makes sets the source dir based on FETCHCONTENT_BASE_DIR
  # which defaults to "${CMAKE_BINARY_DIR}/_deps"; the src and build directories
  # are then set as:
  fetch_content_base_dir = state.workspace.cmake_vars.get(
      "FETCHCONTENT_BASE_DIR"
  )
  if fetch_content_base_dir:
    return pathlib.PurePath(fetch_content_base_dir)
  return state.active_repo.repository.cmake_binary_dir.joinpath("_deps")
