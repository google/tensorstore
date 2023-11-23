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

import argparse
import importlib
import json
import pathlib
import platform
import re
import shlex
from typing import Dict, List, Optional, Set, Tuple, Union

from .cmake_repository import CMakeRepository
from .cmake_target import CMakePackage
from .cmake_target import CMakeTargetPair
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
  options: Dict[str, List[str]] = {}
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
      root_repository: CMakeRepository,
      cmake_vars: Dict[str, str],
      save_workspace: Optional[str] = None,
  ):
    # Known repositories.
    self.root_repository: CMakeRepository = root_repository
    self.all_repositories: Dict[RepositoryId, CMakeRepository] = {}
    self.all_repositories[root_repository.repository_id] = root_repository

    # Variables provided by CMake.
    self.cmake_vars: Dict[str, str] = cmake_vars
    self.save_workspace = save_workspace

    # Maps bazel repo names to CMake project name/prefix.
    self.active_repository: Optional["Repository"] = None
    self.host_platform_name = _PLATFORM_NAME_TO_BAZEL_PLATFORM.get(
        platform.system()
    )
    self.values: Set[Tuple[str, str]] = set()
    self.copts: List[str] = []
    self.cxxopts: List[str] = []
    self.cdefines: List[str] = []
    self.ignored_libraries: Set[TargetId] = set()
    self._modules: Set[str] = set()

    # _persisted_targets persist TargetInfo; these are resolved through
    # EvaluationState.get_optional_target_info() and provide a TargetInfo.
    self._persisted_target_info: Dict[TargetId, TargetInfo] = {}

    # Log level
    self._verbose: int = cmake_logging_verbose_level(
        cmake_vars.get("CMAKE_MESSAGE_LOG_LEVEL")
    )
    if self._verbose > 1:
      print(json.dumps(cmake_vars, sort_keys=True, indent="  "))

  @property
  def verbose(self) -> int:
    return self._verbose

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
  ) -> Optional[str]:
    repo = self.all_repositories.get(target.repository_id, None)
    if repo:
      return repo.cmake_project_name

  def get_persisted_canonical_name(
      self, target: TargetId
  ) -> Optional[CMakeTargetPair]:
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

  def ignore_library(self, target: TargetId) -> None:
    """Marks a bzl library to be ignored.

    When requested via a `load` call, a placeholder object will be returned
    instead.

    Like all `Workspace` state, this also propagates to bazel_to_cmake
    invocations for dependencies.
    """
    assert isinstance(target, TargetId)
    self.ignored_libraries.add(target)

  def load_bazelrc(self, path: str) -> None:
    """Loads options from a `.bazelrc` file.

    This currently only uses `--define` options.
    """
    self.add_bazelrc(_parse_bazelrc(path))

  def add_bazelrc(self, options: Dict[str, List[str]]) -> None:
    """Updates options based on a parsed `.bazelrc` file.

    This currently only uses `--define`, `--copt`, and `--cxxopt` options.
    """
    build_options = []
    build_options.extend(options.get("build", []))
    host_platform_name = self.host_platform_name
    if host_platform_name == "windows" and cmake_is_true(
        self.cmake_vars.get("MINGW")
    ):
      host_platform_name = "windows_x86_64_mingw"
    if host_platform_name is not None:
      build_options.extend(options.get(f"build:{host_platform_name}", []))

    class ConfigAction(argparse.Action):

      def __call__(
          self,  # type: ignore[override]
          parser: argparse.ArgumentParser,
          namespace: argparse.Namespace,
          values: str,
          option_string: Optional[str] = None,
      ):
        parser.parse_known_args(
            options.get(f"build:{values}", []), namespace=namespace
        )

    ap = argparse.ArgumentParser()
    ap.add_argument("--copt", action="append", default=[])
    ap.add_argument("--cxxopt", action="append", default=[])
    ap.add_argument("--per_file_copt", action="append", default=[])
    ap.add_argument("--define", action="append", default=[])
    ap.add_argument("--config", action=ConfigAction)
    args, _ = ap.parse_known_args(build_options)

    def translate_per_file_copt_to_cxxopt():
      # Translate `per_file_copt` options, which are used to workaround
      # https://github.com/bazelbuild/bazel/issues/15550, back to `cxxopt`.
      cxxopt_prefix = r".*\.cc$,.*\.cpp$@"

      for value in args.per_file_copt:
        if value.startswith(cxxopt_prefix):
          args.cxxopt.extend(value[len(cxxopt_prefix) :].split(","))

    translate_per_file_copt_to_cxxopt()

    self.values.update(("define", x) for x in args.define)

    def filter_copts(opts):
      return [
          opt
          for opt in opts
          if re.match("^(?:[-/]std|-fdiagnostics-color=|[-/]D)", opt) is None
      ]

    copts = filter_copts(args.copt)
    cxxopts = filter_copts(args.cxxopt)
    self.copts.extend(copts)
    self.cxxopts.extend(cxxopts)

    seen = set(self.cdefines)
    for opt in args.copt + args.cxxopt:
      if re.match("^(?:[-/]D)", opt) is not None:
        x = opt[2:]
        if x not in seen:
          seen.add(x)
          self.cdefines.append(x)

  def add_module(self, module_name: str):
    self._modules.add(module_name)

  def load_modules(self):
    """Load modules added by add_module."""
    for module_name in self._modules:
      if module_name.startswith("."):
        importlib.import_module(module_name, package=__package__)
      else:
        importlib.import_module(module_name)


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
      repository: CMakeRepository,
      bindings: Dict[TargetId, TargetId],
      top_level: bool,
  ):
    self._workspace: Workspace = workspace
    self._repository: CMakeRepository = repository
    self._bindings: Dict[TargetId, TargetId] = bindings
    self.top_level: bool = top_level

    # Check fidelity of workspace.
    assert repository.repository_id in workspace.all_repositories
    assert workspace.all_repositories[repository.repository_id] == repository

    workspace.active_repository = self
    if workspace.verbose:
      print(f"ActiveRepository: {repr(self)}")

  @property
  def workspace(self) -> Workspace:
    return self._workspace

  @property
  def repository(self) -> CMakeRepository:
    return self._repository

  @property
  def bindings(self) -> Dict[TargetId, TargetId]:
    return self._bindings

  @property
  def repository_id(self) -> RepositoryId:
    return self._repository.repository_id

  @property
  def cmake_project_name(self) -> CMakePackage:
    return self._repository.cmake_project_name

  @property
  def source_directory(self) -> pathlib.PurePath:
    return self._repository.source_directory

  @property
  def cmake_binary_dir(self) -> pathlib.PurePath:
    return self._repository.cmake_binary_dir

  @property
  def repo_mapping(self) -> Dict[RepositoryId, RepositoryId]:
    return self._repository.repo_mapping
