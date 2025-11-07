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
"""Defines data structures representing a Bazel workspace."""

# pylint: disable=missing-function-docstring,relative-beyond-top-level,g-doc-args

import importlib
import json
import pathlib
import platform
import shlex
from typing import Union

from .cmake_repository import CMakeRepository
from .cmake_target import CMakeTargetPair
from .parse_bazelrc import ParsedBazelrc
from .starlark.bazel_target import PackageId
from .starlark.bazel_target import RepositoryId
from .starlark.bazel_target import TargetId
from .starlark.provider import TargetInfo
from .util import cmake_is_true
from .util import cmake_logging_verbose_level

# Maps `platform.system()` value to Bazel host platform name for use in
# .bazelrc.
_PLATFORM_NAME_TO_BAZEL_PLATFORM = {
    "Windows": "windows",
    "FreeBSD": "freebsd",
    "OpenBSD": "openbsd",
    "Darwin": "macos",
    "Linux": "linux",
}


def _parse_bazelrc(path: str):
  options: dict[str, list[str]] = {}
  for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
      continue
    parts = shlex.split(line)
    options.setdefault(parts[0], []).extend(parts[1:])
  return options


class Workspace:
  """Relevant state of entire Bazel workspace.

  This is initialized in the top-level build and loaded via pickle for
  sub-projects that require bazel_to_cmake, in order to allow sub-projects to
  access targets defined by the top-level project.

  The workspace depends on the particular CMake build configuration, which
  allows all `select` expressions to be fully evaluated.
  """

  def __init__(
      self,
      root_repository_id: RepositoryId,
      cmake_vars: dict[str, str],
      save_workspace: str | None = None,
  ):
    # Known repositories.
    self.root_repository_id: RepositoryId = root_repository_id
    self.all_repositories: dict[RepositoryId, CMakeRepository] = {}

    # Variables provided by CMake.
    self.cmake_vars: dict[str, str] = cmake_vars
    self.save_workspace = save_workspace

    # Maps bazel repo names to CMake project name/prefix.
    self.host_platform_name = _PLATFORM_NAME_TO_BAZEL_PLATFORM.get(
        platform.system()
    )
    self._modules: set[str] = set()

    # Create a parser for the .bazelrc files
    host_platform_name = self.host_platform_name
    if host_platform_name == "windows" and cmake_is_true(
        self.cmake_vars.get("MINGW")
    ):
      host_platform_name = "windows_x86_64_mingw"
    self._parsed_bazelrc = ParsedBazelrc(host_platform_name)

    self.global_ignored_libraries: set[TargetId] = set()

    # _persisted_targets persist TargetInfo; these are resolved through
    # EvaluationState.get_optional_target_info() and provide a TargetInfo.
    self._persisted_target_info: dict[TargetId, TargetInfo] = {}

    # Log level
    self._verbose: int = cmake_logging_verbose_level(
        cmake_vars.get("CMAKE_MESSAGE_LOG_LEVEL")
    )
    if self._verbose > 1:
      print(json.dumps(cmake_vars, sort_keys=True, indent="  "))

  def __repr__(self) -> str:
    return (
        "Workspace\n# {\n"
        + "\n".join(f"#   {k}={repr(v)}" for k, v in self.__dict__.items())
        + "\n# }\n"
    )

  @property
  def verbose(self) -> int:
    return self._verbose

  @property
  def values(self) -> set[tuple[str, str]]:
    return self._parsed_bazelrc.values

  @property
  def copts(self) -> list[str]:
    return self._parsed_bazelrc.copts

  @property
  def conlyopts(self) -> list[str]:
    return self._parsed_bazelrc.conlyopts

  @property
  def cxxopts(self) -> list[str]:
    return self._parsed_bazelrc.cxxopts

  @property
  def cdefines(self) -> list[str]:
    return self._parsed_bazelrc.cdefines

  @property
  def linkopts(self) -> list[str]:
    return self._parsed_bazelrc.linkopts

  def get_per_file_copts(self, target: TargetId, src: str) -> list[str]:
    return self._parsed_bazelrc.get_per_file_copts(target, src)

  @property
  def root_repository(self) -> CMakeRepository:
    return self.all_repositories[self.root_repository_id]

  def add_cmake_repository(self, repository: CMakeRepository):
    """Adds a repository to the workspace."""
    assert repository.cmake_project_name

    exists = self.all_repositories.get(repository.repository_id, None)
    if exists:
      assert not exists.source_directory
      assert not exists.cmake_binary_dir
      repository.persisted_canonical_name.update(
          exists.persisted_canonical_name
      )
      repository.repo_mapping.update(exists.repo_mapping)
      if self._verbose:
        print(f"Workspace updating repository {repository.repository_id}")
    elif self._verbose:
      print(
          f"Workspace adding repository {repository.repository_id} =>"
          f" {repository.cmake_project_name}"
      )

    self.all_repositories[repository.repository_id] = repository

  def get_cmake_package_name(
      self, target: Union[RepositoryId, PackageId, TargetId]
  ) -> str | None:
    repo = self.all_repositories.get(target.repository_id, None)
    if repo:
      return repo.cmake_project_name

  def get_persisted_canonical_name(
      self, target: TargetId
  ) -> CMakeTargetPair | None:
    repo = self.all_repositories.get(target.repository_id, None)
    if repo:
      return repo.get_persisted_canonical_name(target)
    return None

  def set_persistent_target_info(self, target: TargetId, info: TargetInfo):
    """Records a persistent mapping from Target to TargetInfo.

    Generally this is used to set global build settings and cmake aliases.
    """
    if target in self._persisted_target_info:
      print(f"Target exists {target} => {info}")
    else:
      if self._verbose > 1:
        print(f"Persisting {target} => {repr(info)}")
      self._persisted_target_info[target] = info

  def load_bazelrc(self, path: str) -> None:
    """Loads options from a `.bazelrc` file.

    This currently only uses `--define` options.
    """
    self._parsed_bazelrc.load_bazelrc(path)
    if self._verbose:
      print(f"Current bazelrc options:\n{self._parsed_bazelrc}")

  def add_bazelrc(self, options: dict[str, list[str]]) -> None:
    """Updates options based on a parsed `.bazelrc` file.

    This currently only uses `--define`, `--copt`, and `--cxxopt` options.
    """
    self._parsed_bazelrc.add_bazelrc(options)
    if self._verbose:
      print(f"Current bazelrc options:\n{self._parsed_bazelrc}")

  def add_module(self, module_name: str):
    self._modules.add(module_name)

  def load_modules(self):
    """Load modules added by add_module."""
    for module_name in self._modules:
      if module_name.startswith("."):
        importlib.import_module(module_name, package=__package__)
      else:
        importlib.import_module(module_name)
