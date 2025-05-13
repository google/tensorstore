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

import pathlib
from typing import Dict, Set

from .cmake_repository import CMakeRepository
from .cmake_target import CMakePackage
from .starlark.bazel_target import RepositoryId
from .starlark.bazel_target import TargetId
from .workspace import Workspace


class Repository:
  """Represents a single Bazel repository and corresponding CMake project.

  This associates the source and output directories with a repository.

  Each run of `bazel_to_cmake` operates on just a single primary repository, but
  may load bzl libraries for the top-level repository as well.

  The `Workspace` holds a mapping of repository names to `Repository` objects.
  In the top-level invocation of `bazel_to_cmake`, only the top-level repository
  is present.  When invoked on dependencies, both the top-level repository and
  the dependency currently being processed are present.
  """

  def __init__(
      self,
      workspace: Workspace,
      repository_id: RepositoryId,
      bindings: Dict[TargetId, TargetId],
      top_level: bool,
  ):
    self._workspace: Workspace = workspace
    self._repository_id: RepositoryId = repository_id
    self._bindings: Dict[TargetId, TargetId] = bindings
    self.top_level: bool = top_level
    self.ignored_libraries: Set[TargetId] = (
        workspace.global_ignored_libraries.copy()
    )
    self._repository = None
    if workspace.verbose:
      print(f"ActiveRepository: {repr(self)}")

  def ignore_library(self, target: TargetId) -> None:
    """Marks a bzl library to be ignored.

    When requested via a `load` call, a placeholder object will be returned
    instead.

    Like all `Workspace` state, this also propagates to bazel_to_cmake
    invocations for dependencies.
    """
    assert isinstance(target, TargetId)
    self.ignored_libraries.add(target)

  @property
  def workspace(self) -> Workspace:
    return self._workspace

  @property
  def repository(self) -> CMakeRepository:
    if self._repository is None:
      if self._repository_id not in self._workspace.all_repositories:
        raise ValueError(f"{self._repository_id} is not available")
      self._repository = self._workspace.all_repositories[self._repository_id]
    return self._repository

  @property
  def bindings(self) -> Dict[TargetId, TargetId]:
    return self._bindings

  @property
  def repository_id(self) -> RepositoryId:
    return self._repository_id

  @property
  def cmake_project_name(self) -> CMakePackage:
    return self.repository.cmake_project_name

  @property
  def source_directory(self) -> pathlib.PurePath:
    return self.repository.source_directory

  @property
  def cmake_binary_dir(self) -> pathlib.PurePath:
    return self.repository.cmake_binary_dir

  @property
  def repo_mapping(self) -> Dict[RepositoryId, RepositoryId]:
    return self.repository.repo_mapping
