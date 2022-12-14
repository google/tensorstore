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
"""Package metadata and Visibility utilities."""

import enum
from typing import List, Optional

from .starlark.bazel_target import TargetId
from .util import cmake_is_true
from .workspace import Repository


class AnalyzeMode(enum.Enum):
  NOTHING = 0
  EVERYTHING = 1
  PUBLIC_ONLY = 2


class Package:
  """Represents a package within a Bazel repository."""

  def __init__(
      self,
      repository: Repository,
      package_name: str,
  ):
    assert repository
    self.repository = repository
    self.package_id = repository.repository_id.get_package_id(package_name)
    self._default_visibility: List[TargetId] = []

    # Whether analyze by default returns only public targets.
    if (repository.top_level and
        cmake_is_true(repository.workspace.cmake_vars["PROJECT_IS_TOP_LEVEL"])):
      self.analyze_mode = AnalyzeMode.EVERYTHING
      self.test_analyze_mode = AnalyzeMode.EVERYTHING
    else:
      self.analyze_mode = AnalyzeMode.PUBLIC_ONLY
      self.test_analyze_mode = AnalyzeMode.NOTHING

  def __repr__(self):
    return f"Package(repository={repr(self.repository)},package_name={self.package_id.package_name})"

  def __str__(self):
    raise AttributeError("Package -> string conversion not allowed")


class Visibility:
  """Manipulates the default visibility of a package.

  In bazel_to_cmake, visibility is used to determine whether to
  add the target to the CMake build by default.
  (See EvaluationState.add_rule analyze_by_default).
  """

  def __init__(self, package: Package):
    self._package = package

  def set_default_visibility(self, visibility: List[TargetId]):
    self._package._default_visibility = visibility  # pylint: disable=protected-access

  def analyze_by_default(self, visibility: Optional[List[TargetId]]):
    return (self._package.analyze_mode == AnalyzeMode.EVERYTHING) or (
        self._package.analyze_mode == AnalyzeMode.PUBLIC_ONLY and
        self.is_public(visibility))

  def analyze_test_by_default(self, visibility: Optional[List[TargetId]]):
    # Treat tests as private regardless of actual visibility.
    del visibility
    return self._package.test_analyze_mode == AnalyzeMode.EVERYTHING

  def is_public(self, visibility: Optional[List[TargetId]]):
    if visibility is not None:
      vis_targets = visibility
    else:
      vis_targets = self._package._default_visibility  # pylint: disable=protected-access
    for target in vis_targets:
      if target.package_name == "visibility" and target.target_name == "public":
        # //visibility:public
        return True
    return False
