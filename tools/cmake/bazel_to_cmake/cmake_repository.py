# Copyright 2023 The TensorStore Authors
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
"""CMakeRepository maintains persistent information about CMake Repositories."""

# pylint: disable=missing-function-docstring

import hashlib
import pathlib
import re
from typing import Any, Dict, List, NamedTuple, Optional

from .cmake_target import CMakePackage
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetPair
from .starlark.bazel_target import apply_repo_mapping
from .starlark.bazel_target import RepositoryId
from .starlark.bazel_target import TargetId

_SPLIT_RE = re.compile("[:/]+")
_BIG = 35


class CMakeRepository(NamedTuple):
  """Represents a single Bazel repository and corresponding CMake project."""

  repository_id: RepositoryId
  cmake_project_name: CMakePackage
  source_directory: pathlib.PurePath
  cmake_binary_dir: pathlib.PurePath
  repo_mapping: Dict[RepositoryId, RepositoryId]
  persisted_canonical_name: Dict[TargetId, CMakeTargetPair]

  def get_source_file_path(self, target_id: TargetId) -> pathlib.PurePath:
    assert self.source_directory
    return self.source_directory.joinpath(
        target_id.package_name, target_id.target_name
    )

  def get_generated_file_path(self, target_id: TargetId) -> pathlib.PurePath:
    assert self.cmake_binary_dir
    return self.cmake_binary_dir.joinpath(
        target_id.package_name, target_id.target_name
    )

  def get_cmake_target_pair(self, target_id: TargetId) -> CMakeTargetPair:
    assert target_id.repository_id == self.repository_id
    persisted = self.persisted_canonical_name.get(target_id)
    if persisted is not None:
      return persisted
    return label_to_generated_cmake_target(target_id, self.cmake_project_name)

  def apply_repo_mapping(self, target_id: TargetId) -> TargetId:
    """Apply a repository mapping to a TargetId."""
    return apply_repo_mapping(target_id, self.repo_mapping)

  def set_persisted_canonical_name(
      self,
      target_id: TargetId,
      cmake_target_pair: CMakeTargetPair,
  ):
    """Records a persistent mapping from Target to CMakeTargetPair.

    set_persisted_canonical_name allows a persistent pre-defined mapping from
    bazel target to a cmake name/alias. This allows overriding the name
    generated automatically by bazel_to_cmake, and is set from the
    third_party_http_archive target_mapping setting.
    """
    assert cmake_target_pair.cmake_package is not None
    assert cmake_target_pair.cmake_package == self.cmake_project_name
    assert target_id.repository_id == self.repository_id
    self.persisted_canonical_name[target_id] = cmake_target_pair

  def get_persisted_canonical_name(
      self, target_id: TargetId
  ) -> Optional[CMakeTargetPair]:
    assert target_id.repository_id == self.repository_id
    return self.persisted_canonical_name.get(target_id, None)


def make_repo_mapping(
    repository_id: RepositoryId, repo_mapping: Any
) -> Dict[RepositoryId, RepositoryId]:
  def get_pairs():
    if isinstance(repo_mapping, dict):
      return repo_mapping.items()
    return repo_mapping

  # Add repo mappings
  output: Dict[RepositoryId, RepositoryId] = {}
  for x, y in get_pairs():
    if not isinstance(x, RepositoryId):
      x = str(x)
      assert x.startswith("@")
      x = RepositoryId(x[1:])
    assert x != repository_id
    if not isinstance(y, RepositoryId):
      y = str(y)
      assert y.startswith("@")
      y = RepositoryId(y[1:])
    output[x] = y
  return output


def label_to_generated_cmake_target(
    target_id: TargetId, cmake_project: CMakePackage
) -> CMakeTargetPair:
  """Computes the generated CMake target corresponding to a Bazel target."""

  parts: List[str] = []
  parts.extend(x for x in _SPLIT_RE.split(target_id.package_name) if x)
  parts.extend(x for x in _SPLIT_RE.split(target_id.target_name) if x)
  if parts[0].lower() == str(cmake_project).lower() and len(parts) > 1:
    parts = parts[1:]

  if len(parts) >= 2 and parts[-1] == parts[-2]:
    parts = parts[:-1]

  # CMake cannot handle paths > 250 bytes, so rewrite long targets.
  if len(parts) > 2 and sum(len(x) for x in parts[:-1]) > _BIG:
    m = hashlib.sha256()
    m.update(bytes(target_id.package_name, "utf-8"))
    m.update(bytes(target_id.target_name, "utf-8"))
    target_name = "_".join([
        parts[0],
        m.hexdigest().lower()[:8],
        "".join([x[0] for x in parts[1:]]),
    ])
  else:
    target_name = "_".join(parts)

  return CMakeTargetPair(
      cmake_project,
      CMakeTarget(f"{cmake_project}_{target_name}"),
      CMakeTarget(f"{cmake_project}::{target_name}"),
  )
