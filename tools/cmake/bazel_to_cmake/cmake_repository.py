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

from collections.abc import Iterable
import hashlib
import pathlib
import re
from typing import Any, NamedTuple

from .cmake_target import CMakePackage
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetPair
from .starlark.bazel_target import apply_repo_mapping
from .starlark.bazel_target import RepositoryId
from .starlark.bazel_target import TargetId
from .util import make_relative_path
from .util import PathLike

_SPLIT_RE = re.compile("[:/]+")
_BIG = 35

PROJECT_SOURCE_DIR = "${PROJECT_SOURCE_DIR}"
PROJECT_BINARY_DIR = "${PROJECT_BINARY_DIR}"


class CMakeRepository(NamedTuple):
  """Represents a single Bazel repository and corresponding CMake project."""

  repository_id: RepositoryId
  cmake_project_name: CMakePackage
  source_directory: pathlib.PurePath
  cmake_binary_dir: pathlib.PurePath
  repo_mapping: dict[RepositoryId, RepositoryId]
  persisted_canonical_name: dict[TargetId, CMakeTargetPair]
  executable_targets: set[TargetId]

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
  ) -> None:
    """Records a persistent mapping from Target to CMakeTargetPair.

    set_persisted_canonical_name allows a persistent pre-defined mapping from
    bazel target to a cmake name/alias. This allows overriding the name
    generated automatically by bazel_to_cmake, and is set from the
    third_party_http_archive target_mapping setting.
    """
    assert isinstance(
        cmake_target_pair, CMakeTargetPair
    ), f"{repr(cmake_target_pair)}"
    assert isinstance(
        cmake_target_pair.cmake_package, CMakePackage
    ), f"{repr(cmake_target_pair)}"

    assert cmake_target_pair.cmake_package is not None
    assert cmake_target_pair.cmake_package == self.cmake_project_name
    assert target_id.repository_id == self.repository_id
    self.persisted_canonical_name[target_id] = cmake_target_pair

  def get_persisted_canonical_name(
      self, target_id: TargetId
  ) -> CMakeTargetPair | None:
    assert target_id.repository_id == self.repository_id
    return self.persisted_canonical_name.get(target_id, None)

  def replace_with_cmake_macro_dirs(
      self, paths: Iterable[PathLike]
  ) -> list[str]:
    """Substitute reposotory path prefixes with CMake PROJECT_{*}_DIR macros."""
    result: list[str] = []
    for x in paths:
      c, relative_path = make_relative_path(
          x,
          (PROJECT_SOURCE_DIR, self.source_directory),
          (PROJECT_BINARY_DIR, self.cmake_binary_dir),
      )
      if c is None:
        result.append(str(relative_path))
      elif relative_path.as_posix() == ".":
        result.append(c)
      else:
        result.append(f"{c}/{relative_path.as_posix()}")
    return result

  def with_cmake_directories(
      self,
      source_directory: pathlib.PurePath | None = None,
      cmake_binary_dir: pathlib.PurePath | None = None,
  ) -> "CMakeRepository":
    """Returns a CMakeRepository with new source and binary directories.

    Args:
      source_directory: The new source directory.
      cmake_binary_dir: The new CMake binary directory.
    """
    return self._replace(
        source_directory=source_directory,
        cmake_binary_dir=cmake_binary_dir,
    )

  @classmethod
  def from_config(
      cls,
      repository_id: RepositoryId,
      config: dict[str, Any],
      source_directory: pathlib.PurePath | None = None,
      cmake_binary_dir: pathlib.PurePath | None = None,
  ) -> "CMakeRepository":
    """Creates a CMakeRepository from a configuration dictionary.

    Args:
      repository_id: The Bazel RepositoryId.
      config: A dictionary containing the repository configuration. Expected
        keys include "cmake_project_name", "cmake_name", "cmake_target_mapping",
        "executable_targets", and "repo_mapping".
      source_directory: The source directory for the CMake project.
      cmake_binary_dir: The binary directory for the CMake build.

    Returns:
      A new CMakeRepository instance.
    """
    cmake_project_name = config.get(
        "cmake_name",
        config.get("cmake_project_name", repository_id.repository_name),
    )
    assert cmake_project_name is not None

    persisted_canonical_name = {}
    for label, target in config.get("cmake_target_mapping", {}).items():
      target_id = repository_id.parse_target(label)
      persisted_canonical_name[target_id] = CMakeTargetPair(
          CMakePackage(cmake_project_name),
          CMakeTarget(target.replace("::", "_")),
          CMakeTarget(target),
      )

    executable_targets: set[TargetId] = set()
    for label in config.get("executable_targets", []):
      executable_targets.add(repository_id.parse_target(label))

    repo_mapping = make_repo_mapping(
        repository_id, config.get("repo_mapping", {})
    )
    return cls(
        repository_id=repository_id,
        cmake_project_name=CMakePackage(cmake_project_name),
        source_directory=source_directory,
        cmake_binary_dir=cmake_binary_dir,
        repo_mapping=repo_mapping,
        persisted_canonical_name=persisted_canonical_name,
        executable_targets=executable_targets,
    )


def make_repo_mapping(
    repository_id: RepositoryId, repo_mapping: Any
) -> dict[RepositoryId, RepositoryId]:
  def get_pairs():
    if isinstance(repo_mapping, dict):
      return repo_mapping.items()
    return repo_mapping

  # Add repo mappings
  output: dict[RepositoryId, RepositoryId] = {}
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
  assert isinstance(cmake_project, CMakePackage), f"{repr(cmake_project)}"

  parts: list[str] = []
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
