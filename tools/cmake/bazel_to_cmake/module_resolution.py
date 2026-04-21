# Copyright 2026 The TensorStore Authors
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
"""Logic for Bzlmod module resolution and repository setup."""

# pylint: disable=g-importing-member,g-doc-args
# pylint: disable=missing-function-docstring,unused-argument,protected-access

import collections
import io
import json
import pathlib
import traceback
from typing import Any

from .bzl_library import helpers
from .bzl_library import third_party_http_archive
from .cmake_repository import CMakeRepository
from .evaluation_state import EvaluationState
from .evaluation_state import get_fetch_content_base_dir
from .starlark.bazel_target import RepositoryId
from .starlark.exec import compile_and_exec
from .starlark.label import RelativeLabel


def _cmake_repository_from_directory(
    repository_id: RepositoryId,
    directory: pathlib.Path,
    registry_path: pathlib.Path | None = None,
    fetch_content_base_dir: pathlib.Path | None = None,
) -> tuple[CMakeRepository, dict[str, Any]]:
  """Loads a CMakeRepository from a directory.

  This searches for `bazel_to_cmake.json` and `source.json` in the specified
  directory.

  Returns:
    A tuple of the CMakeRepository and the configuration dictionary loaded from
    `bazel_to_cmake.json`.
  """
  config = {}
  config_path = directory.joinpath("bazel_to_cmake.json")
  if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
      config = json.load(f)

  source_directory: pathlib.PurePath | None = None
  source_json_path = directory.joinpath("source.json")
  if source_json_path.exists():
    with open(source_json_path, "r", encoding="utf-8") as f:
      source_config = json.load(f)
      if source_config.get("type") == "local_path":
        source_directory = pathlib.Path(source_config["path"])
        if not source_directory.is_absolute() and registry_path:
          source_directory = registry_path.joinpath(source_directory)

  repo = CMakeRepository.from_config(repository_id, config, source_directory)

  if fetch_content_base_dir:
    source_dir = repo.source_directory
    if source_dir is None:
      source_dir = fetch_content_base_dir.joinpath(
          f"{repo.cmake_project_name.lower()}-src"
      )
    binary_dir = fetch_content_base_dir.joinpath(
        f"{repo.cmake_project_name.lower()}-build"
    )
    repo = repo.with_cmake_directories(
        source_directory=source_dir,
        cmake_binary_dir=binary_dir,
    )

  return repo, config


class ModuleResolver:
  """Handles MODULE.bazel resolution and overrides.

  This class translates Bazel module dependencies into CMake-compatible
  FetchContent or find_package configurations by searching for metadata
  in local registries (typically third_party/modules).

  The call path for these methods is:
    scope_module_file.py -> evaluation_impl.py -> module_resolution.py

  All of these are WIP methods.
  """

  def __init__(self, state: EvaluationState):
    self._state = state
    self._module_name: str | None = None
    self._module_version: str | None = None
    self._module_overrides: dict[str, dict[str, Any]] = {}
    self._lockfile_repos: dict[str, dict[str, Any]] = collections.defaultdict(
        dict
    )

  def load_lockfile(self, lockfile_json: dict[str, Any]) -> None:
    """Loads resolved repositories from MODULE.bazel.lock."""
    module_extensions = lockfile_json.get("moduleExtensions", {})
    for ext_id, ext_data in module_extensions.items():
      general = ext_data.get("general", {})
      generated_repos = general.get("generatedRepoSpecs", {})
      for repo_name, repo_spec in generated_repos.items():
        self._lockfile_repos[ext_id][repo_name] = repo_spec

  def use_repo(self, extension_proxy, *args, **kwargs) -> None:
    bzl_file = extension_proxy.bzl_file
    ext_name = extension_proxy.name

    path_part = bzl_file
    if bzl_file.startswith("@"):
      parts = bzl_file.split("//", 1)
      if len(parts) == 2:
        path_part = parts[1]

    matching_ext_id = None
    for ext_id in self._lockfile_repos:
      if ext_id.endswith(f"%{ext_name}") and path_part in ext_id:
        matching_ext_id = ext_id
        break

    if not matching_ext_id:
      print(
          "Warning: Could not find lockfile entry for extension"
          f" {bzl_file}%{ext_name}"
      )
      return

    generated_repos = self._lockfile_repos[matching_ext_id]

    for repo_name in args:
      self._use_repo_impl(repo_name, repo_name, generated_repos)

    for local_name, remote_name in kwargs.items():
      self._use_repo_impl(local_name, remote_name, generated_repos)

  def _use_repo_impl(
      self, local_name: str, remote_name: str, generated_repos: dict[str, Any]
  ) -> None:
    repo_spec = generated_repos.get(remote_name)
    if not repo_spec:
      print(f"Warning: Could not find repo {remote_name} in extension results")
      return

    repo_rule_id = repo_spec.get("repoRuleId", "")
    attributes = repo_spec.get("attributes", {})

    repository_id = RepositoryId(local_name)
    repo = self._state.workspace.all_repositories.get(repository_id)
    if not repo:
      repo = CMakeRepository(
          repository_id=repository_id,
          cmake_project_name=local_name,
          source_directory=pathlib.PurePath(),
          cmake_binary_dir=pathlib.PurePath(),
          repo_mapping={},
          persisted_canonical_name={},
          executable_targets=set(),
      )
      self._state.workspace.add_cmake_repository(repo)

    fetch_content_base_dir = get_fetch_content_base_dir(self._state)
    repo = repo._replace(
        source_directory=fetch_content_base_dir.joinpath(
            f"{local_name.lower()}-src"
        ),
        cmake_binary_dir=fetch_content_base_dir.joinpath(
            f"{local_name.lower()}-build"
        ),
    )
    self._state.workspace.all_repositories[repository_id] = repo

    if "http_archive" in repo_rule_id:
      urls = attributes.get("urls")
      if not urls and "url" in attributes:
        urls = [attributes["url"]]
      kwargs = {
          "urls": urls,
          "strip_prefix": attributes.get("strip_prefix"),
          "patch_cmds": attributes.get("patch_cmds"),
          "patches": attributes.get("patches"),
          "sha256": attributes.get("sha256"),
      }
      third_party_http_archive._emit_fetch_content_impl(
          self._state.evaluation_context,
          repo,
          name=local_name,
          cmake_name=local_name,
          **kwargs,
      )
    else:
      print(f"Warning: Unsupported repo rule {repo_rule_id} for {local_name}")

  def set_module_name_version(self, name: str, version: str) -> None:
    """Sets the current module name and version.

    Corresponds to the `module()` function in `MODULE.bazel`.
    """
    self._module_name = name
    self._module_version = version

  def add_bazel_dep(
      self,
      name: str,
      version: str,
      max_compatibility_level: int,
      repo_name: str,
      dev_dependency: bool,
  ) -> None:
    """Adds a Bazel dependency.

    Corresponds to the `bazel_dep()` function in `MODULE.bazel`.
    """
    if dev_dependency:
      return
    actual_repo_name = repo_name or name
    repository_id = RepositoryId(actual_repo_name)

    if repository_id in self._state.workspace.all_repositories:
      return

    # 1. Check for explicit overrides (e.g. local_path_override in MODULE.bazel)
    override = self._module_overrides.get(name)
    if override:
      self._apply_module_override(name, version, actual_repo_name, override)
      return

    # 2. Check external config (legacy workspace-style definitions)
    ext_config = getattr(
        self._state.workspace, "_external_repo_configs", {}
    ).get(RepositoryId(name))
    if ext_config:
      self._apply_external_config(name, actual_repo_name, ext_config)
      return

    # 3. Registry lookup: Search for bazel_to_cmake.json in third_party/modules
    self._resolve_from_registries(name, version, actual_repo_name)

  def add_module_override(
      self, module_name: str, override_info: dict[str, Any]
  ) -> None:
    """Adds a module override.

    Corresponds to various override functions in `MODULE.bazel`, such as
    `local_path_override`, `archive_override`, `git_override`, and
    `single_version_override`.
    """
    self._module_overrides[module_name] = override_info
    if override_info["type"] == "local_path":
      self._set_local_path_override(module_name, override_info["path"])

  def _set_local_path_override(self, module_name: str, path: str):
    repository_id = RepositoryId(module_name)
    source_directory = pathlib.Path(path)
    if not source_directory.is_absolute():
      source_directory = self._state.active_repo.source_directory.joinpath(path)

    repo = self._state.workspace.all_repositories.get(repository_id)
    if repo:
      self._state.workspace.add_cmake_repository(
          repo._replace(source_directory=source_directory)
      )

  def _apply_module_override(self, name, version, actual_repo_name, override):
    if override["type"] == "local_path":
      self._set_local_path_override(name, override["path"])
    elif override["type"] == "archive":
      repository_id = RepositoryId(actual_repo_name)
      repo = self._state.workspace.all_repositories.get(repository_id)
      if not repo:
        repo = CMakeRepository(
            repository_id=repository_id,
            cmake_project_name=name,
            source_directory=pathlib.PurePath(),
            cmake_binary_dir=pathlib.PurePath(),
            repo_mapping={},
            persisted_canonical_name={},
            executable_targets=set(),
        )
        self._state.workspace.add_cmake_repository(repo)

      fetch_content_base_dir = get_fetch_content_base_dir(self._state)
      cmake_name = name
      repo = repo._replace(
          source_directory=fetch_content_base_dir.joinpath(
              f"{cmake_name.lower()}-src"
          ),
          cmake_binary_dir=fetch_content_base_dir.joinpath(
              f"{cmake_name.lower()}-build"
          ),
      )
      self._state.workspace.all_repositories[repository_id] = repo

      kwargs = {
          "urls": override.get("urls"),
          "strip_prefix": override.get("strip_prefix"),
          "patch_cmds": override.get("patch_cmds"),
          "patches": override.get("patches"),
          "sha256": override.get("integrity"),
      }
      third_party_http_archive._emit_fetch_content_impl(
          self._state.evaluation_context,
          repo,
          name=name,
          cmake_name=cmake_name,
          **kwargs,
      )
    elif override["type"] == "git":
      repository_id = RepositoryId(actual_repo_name)
      repo = self._state.workspace.all_repositories.get(repository_id)
      if not repo:
        repo = CMakeRepository(
            repository_id=repository_id,
            cmake_project_name=name,
            source_directory=pathlib.PurePath(),
            cmake_binary_dir=pathlib.PurePath(),
            repo_mapping={},
            persisted_canonical_name={},
            executable_targets=set(),
        )
        self._state.workspace.add_cmake_repository(repo)

      fetch_content_base_dir = get_fetch_content_base_dir(self._state)
      cmake_name = name
      repo = repo._replace(
          source_directory=fetch_content_base_dir.joinpath(
              f"{cmake_name.lower()}-src"
          ),
          cmake_binary_dir=fetch_content_base_dir.joinpath(
              f"{cmake_name.lower()}-build"
          ),
      )
      self._state.workspace.all_repositories[repository_id] = repo

      kwargs = {
          "git_repository": override.get("remote"),
          "git_tag": override.get("commit") or override.get("tag"),
          "patch_cmds": override.get("patch_cmds"),
          "patches": override.get("patches"),
      }
      third_party_http_archive._emit_fetch_content_impl(
          self._state.evaluation_context,
          repo,
          name=name,
          cmake_name=cmake_name,
          **kwargs,
      )

  def _apply_external_config(self, name, actual_repo_name, ext_config):
    repository_id = RepositoryId(actual_repo_name)
    repo = self._state.workspace.all_repositories.get(repository_id)

    if repo and ("urls" in ext_config or "url" in ext_config):
      fetch_content_base_dir = get_fetch_content_base_dir(self._state)
      cmake_name = ext_config.get("cmake_project_name", name)
      repo = repo._replace(
          source_directory=fetch_content_base_dir.joinpath(
              f"{cmake_name.lower()}-src"
          ),
          cmake_binary_dir=fetch_content_base_dir.joinpath(
              f"{cmake_name.lower()}-build"
          ),
      )
      self._state.workspace.all_repositories[repository_id] = repo

      if repository_id in self._state.workspace.exclude_repositories:
        ext_config = ext_config.copy()
        ext_config.pop("bazel_to_cmake", None)
        ext_config.pop("cmake_target_mapping", None)

      third_party_http_archive._emit_fetch_content_impl(
          self._state.evaluation_context,
          repo,
          name=name,
          cmake_name=ext_config.get("cmake_project_name", name),
          **ext_config,
      )

  def _resolve_from_registries(self, name, version, actual_repo_name):
    """Resolves a module by searching through configured registries."""
    workspace_root = self._state.active_repo.source_directory
    registries = self._state.workspace._parsed_bazelrc.registries or [
        workspace_root.joinpath("third_party").as_posix()
    ]

    for registry in registries:
      if self._try_resolve_from_registry(
          registry, name, version, actual_repo_name
      ):
        return

  def _try_resolve_from_registry(
      self, registry, name, version, actual_repo_name
  ):
    """Attempts to resolve a module from a specific registry."""
    # Only local registries are supported for now.
    if not registry.startswith("http"):
      registry_path = pathlib.Path(registry)
      module_dir = registry_path.joinpath("modules", name, version)
      module_config_path = module_dir.joinpath("bazel_to_cmake.json")
      source_json_path = module_dir.joinpath("source.json")

      if module_config_path.exists() or source_json_path.exists():
        self._load_module_config(
            module_dir,
            registry_path,
            name,
            version,
            actual_repo_name,
        )
        return True
    return False

  def _load_module_config(
      self, module_path, registry_path, name, version, actual_repo_name
  ):
    repository_id = RepositoryId(actual_repo_name)
    fetch_content_base_dir = get_fetch_content_base_dir(self._state)
    try:
      new_repository, config = _cmake_repository_from_directory(
          repository_id=repository_id,
          directory=module_path,
          registry_path=registry_path,
          fetch_content_base_dir=fetch_content_base_dir,
      )
      self._state.workspace.add_cmake_repository(new_repository)

      cmake_project_name = str(new_repository.cmake_project_name)
      bazel_to_cmake_config = config.get("bazel_to_cmake")
      if new_repository.source_directory and bazel_to_cmake_config is not None:
        new_cmakelists = io.StringIO()
        helpers.write_bazel_to_cmake_cmakelists(
            _context=self._state.evaluation_context,
            _new_cmakelists=new_cmakelists,
            _patch_commands=[],
            name=actual_repo_name,
            cmake_name=cmake_project_name,
            bazel_to_cmake=bazel_to_cmake_config,
            cmake_target_mapping=config.get("cmake_target_mapping"),
            repo_mapping=config.get("repo_mapping"),
            build_file=self._state.evaluation_context.resolve_target_or_label(
                bazel_to_cmake_config.get("build_file")
            )
            if bazel_to_cmake_config.get("build_file")
            else None,
            cmake_extra_build_file=self._state.evaluation_context.resolve_target_or_label(
                bazel_to_cmake_config.get("cmake_extra_build_file")
            )
            if bazel_to_cmake_config.get("cmake_extra_build_file")
            else None,
            is_local=True,
        )
        cmakelists_path = pathlib.Path(
            new_repository.source_directory
        ).joinpath("CMakeLists.txt")
        try:
          cmakelists_path.parent.mkdir(parents=True, exist_ok=True)
          cmakelists_path.write_text(
              new_cmakelists.getvalue(), encoding="utf-8"
          )
        except OSError as e:
          print(f"Warning: Failed to write {cmakelists_path}: {e}")

      # Emits the CMake FetchContent code to download/link the dependency.
      third_party_http_archive._emit_fetch_content_impl(
          self._state.evaluation_context,
          new_repository,
          SOURCE_DIR=new_repository.source_directory,
          cmake_name=cmake_project_name,
          name=actual_repo_name,
          bazel_to_cmake=bazel_to_cmake_config,
          cmake_target_mapping=config.get("cmake_target_mapping"),
          cmake_enable_system_package=config.get(
              "cmake_enable_system_package", True
          ),
      )
    except Exception as e:
      print(f"Warning: Failed to process module {name}: {e}")
      traceback.print_exc()

  def include_module_file(self, label: RelativeLabel, scope: Any) -> None:
    """Includes another Starlark file into the `MODULE.bazel` evaluation.

    Corresponds to the `include()` function in `MODULE.bazel`.

    raises:
      FileNotFoundError: If the resolved include file does not exist.
    """
    target_id = self._state.evaluation_context.resolve_target_or_label(label)
    path = pathlib.Path(
        self._state.evaluation_context.workspace_root_for_label(
            target_id.repository_id
        )
    ).joinpath(target_id.package_name, target_id.target_name)
    if not path.exists():
      raise FileNotFoundError(f"Included file not found: {path} (from {label})")

    with open(path, "r", encoding="utf-8") as f:
      content = f.read()

    compile_and_exec(content, str(path), scope)
