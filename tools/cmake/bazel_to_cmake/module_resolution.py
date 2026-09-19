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

import base64
import binascii
import collections
import dataclasses
import io
import json
import os
import pathlib
import shutil
import traceback
from typing import Any
import urllib.parse

from .bzl_library import helpers
from .bzl_library import third_party_http_archive
from .cmake_repository import CMakeRepository
from .cmake_target import CMakePackage
from .evaluation_state import EvaluationState
from .evaluation_state import get_fetch_content_base_dir
from .starlark.bazel_target import RepositoryId
from .starlark.exec import compile_and_exec
from .starlark.label import RelativeLabel
from .util import quote_path


@dataclasses.dataclass
class ResolvedModuleSpec:
  """Normalized specification for a resolved module dependency."""

  repository_id: RepositoryId
  cmake_project_name: str
  source_type: str  # "archive", "git", "local_path"
  source_directory: pathlib.PurePath | None = None
  config: dict[str, Any] = dataclasses.field(default_factory=dict)
  overlay_dir: pathlib.Path | None = None


def _update_config_from_source_json(
    config: dict[str, Any],
    source_config: dict[str, Any],
    directory: pathlib.Path,
    workspace_root: pathlib.Path | None,
):
  if "url" in source_config:
    config["urls"] = [source_config["url"]]
  if "integrity" in source_config:
    val = source_config["integrity"]
    if val.startswith("sha256-"):
      # Pad if necessary
      padding = len(val[7:]) % 4
      b64_str = val[7:] + ("=" * ((4 - padding) % 4))
      try:
        config["sha256"] = binascii.hexlify(base64.b64decode(b64_str)).decode(
            "utf-8"
        )
      except Exception:
        pass
    else:
      config["sha256"] = val
  if "strip_prefix" in source_config:
    config["strip_prefix"] = source_config["strip_prefix"]
  if "patches" in source_config:
    patches = []
    for patch in source_config["patches"].keys():
      if workspace_root and directory.is_relative_to(workspace_root):
        patch_abs = directory.joinpath(patch)
        if (
            not patch_abs.exists()
            and directory.joinpath("patches", patch).exists()
        ):
          patch_abs = directory.joinpath("patches", patch)
        pkg_path = patch_abs.parent.relative_to(workspace_root).as_posix()
        target_name = patch_abs.name
        patches.append(f"//{pkg_path}:{target_name}")
      else:
        patches.append(patch)
    config["patches"] = patches
  if "patch_cmds" in source_config:
    config["patch_cmds"] = source_config["patch_cmds"]
  if "patch_strip" in source_config:
    config["patch_args"] = ["-p" + str(source_config["patch_strip"])]
  if "overlay" in source_config:
    overlay_dir = directory.joinpath("overlay")
    if overlay_dir.exists():
      if "patch_cmds" not in config:
        config["patch_cmds"] = []
      for overlay_file in source_config["overlay"].keys():
        src_f = overlay_dir.joinpath(overlay_file)
        if src_f.exists():
          dst_rel = overlay_file
          config["patch_cmds"].append(
              "${CMAKE_COMMAND} -E make_directory"
              f" {os.path.dirname(dst_rel) or '.'}"
          )
          config["patch_cmds"].append(
              "${CMAKE_COMMAND} -E copy"
              f" {quote_path(src_f)} {quote_path(dst_rel)}"
          )


def _resolve_spec_from_directory(
    repository_id: RepositoryId,
    directory: pathlib.Path,
    registry_path: pathlib.Path | None = None,
    workspace_root: pathlib.Path | None = None,
) -> ResolvedModuleSpec:
  """Resolves a module specification from a directory containing metadata."""
  config = {}
  config_path = directory.joinpath("bazel_to_cmake.json")
  if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
      config = json.load(f)

  source_directory: pathlib.PurePath | None = None
  source_type = "archive"
  source_json_path = directory.joinpath("source.json")
  if source_json_path.exists():
    with open(source_json_path, "r", encoding="utf-8") as f:
      source_config = json.load(f)
      if source_config.get("type") == "local_path":
        source_type = "local_path"
        source_directory = pathlib.Path(source_config["path"])
        if not source_directory.is_absolute() and registry_path:
          source_directory = registry_path.joinpath(source_directory)
      else:
        _update_config_from_source_json(
            config, source_config, directory, workspace_root
        )

  # Normalize relative build files to absolute repository labels
  for k in ("build_file", "system_build_file", "cmake_extra_build_file"):
    if k in config and config[k]:
      v = config[k]
      if (
          isinstance(v, str)
          and not v.startswith("//")
          and not v.startswith("@")
      ):
        v_abs = directory.joinpath(v)
        if workspace_root and v_abs.is_relative_to(workspace_root):
          pkg = v_abs.parent.relative_to(workspace_root).as_posix()
          config[k] = f"//{pkg}:{v_abs.name}"

  cmake_project_name = config.get(
      "cmake_name",
      config.get("cmake_project_name", repository_id.repository_name),
  )

  overlay_dir = directory.joinpath("overlay")
  return ResolvedModuleSpec(
      repository_id=repository_id,
      cmake_project_name=cmake_project_name,
      source_type=source_type,
      source_directory=source_directory,
      config=config,
      overlay_dir=overlay_dir if overlay_dir.exists() else None,
  )


def _cmake_repository_from_directory(
    repository_id: RepositoryId,
    directory: pathlib.Path,
    registry_path: pathlib.Path | None = None,
    fetch_content_base_dir: pathlib.Path | None = None,
    workspace_root: pathlib.Path | None = None,
) -> tuple[CMakeRepository, dict[str, Any]]:
  """Loads a CMakeRepository from a directory.

  This searches for `bazel_to_cmake.json` and `source.json` in the specified
  directory.

  Returns:
    A tuple of the CMakeRepository and the configuration dictionary loaded from
    `bazel_to_cmake.json`.
  """
  spec = _resolve_spec_from_directory(
      repository_id=repository_id,
      directory=directory,
      registry_path=registry_path,
      workspace_root=workspace_root,
  )
  repo = CMakeRepository.from_config(
      repository_id, spec.config, spec.source_directory
  )

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

  return repo, spec.config


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

  def _resolve_lockfile_repo(
      self, local_name: str, remote_name: str, generated_repos: dict[str, Any]
  ) -> ResolvedModuleSpec | None:
    repo_spec = generated_repos.get(remote_name)
    if not repo_spec:
      print(f"Warning: Could not find repo {remote_name} in extension results")
      return None

    repo_rule_id = repo_spec.get("repoRuleId", "")
    attributes = repo_spec.get("attributes", {})

    if "http_archive" not in repo_rule_id:
      print(f"Warning: Unsupported repo rule {repo_rule_id} for {local_name}")
      return None

    urls = attributes.get("urls")
    if not urls and "url" in attributes:
      urls = [attributes["url"]]
    config = {
        "cmake_name": local_name,
        "urls": urls,
        "strip_prefix": attributes.get("strip_prefix"),
        "patch_cmds": attributes.get("patch_cmds"),
        "patches": attributes.get("patches"),
        "sha256": attributes.get("sha256"),
    }
    return ResolvedModuleSpec(
        repository_id=RepositoryId(local_name),
        cmake_project_name=local_name,
        source_type="archive",
        config=config,
    )

  def _use_repo_impl(
      self, local_name: str, remote_name: str, generated_repos: dict[str, Any]
  ) -> None:
    spec = self._resolve_lockfile_repo(
        local_name, remote_name, generated_repos
    )
    if spec:
      self._materialize_module_repo(spec)

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

    spec = self._resolve_dep(name, version, actual_repo_name)
    if spec:
      self._materialize_module_repo(spec)

  def _resolve_dep(
      self, name: str, version: str, actual_repo_name: str
  ) -> ResolvedModuleSpec | None:
    """Resolves a dependency via overrides, external configs, or registries."""
    # 1. Check for explicit overrides (e.g. local_path_override in MODULE.bazel)
    override = self._module_overrides.get(name)
    if override:
      spec = self._resolve_module_override(
          name, version, actual_repo_name, override
      )
      if spec:
        return spec

    # 2. Check external config (legacy workspace-style definitions)
    ext_config = getattr(
        self._state.workspace, "_external_repo_configs", {}
    ).get(RepositoryId(name))
    if ext_config:
      return self._resolve_external_config(name, actual_repo_name, ext_config)

    # 3. Registry lookup: Search for bazel_to_cmake.json in registries
    return self._resolve_from_registries(name, version, actual_repo_name)

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

  def _apply_override_patches_to_spec(
      self, spec: ResolvedModuleSpec, override: dict[str, Any]
  ) -> None:
    if override.get("patches"):
      existing = spec.config.get("patches", [])
      spec.config["patches"] = list(existing) + list(override["patches"])
    if override.get("patch_cmds"):
      existing = spec.config.get("patch_cmds", [])
      spec.config["patch_cmds"] = list(existing) + list(override["patch_cmds"])
    if override.get("patch_strip"):
      spec.config["patch_args"] = ["-p" + str(override["patch_strip"])]

  def _resolve_module_override(
      self,
      name: str,
      version: str,
      actual_repo_name: str,
      override: dict[str, Any],
  ) -> ResolvedModuleSpec | None:
    override_type = override.get("type")
    if override_type == "single":
      version = override.get("version") or version
      override_registry = override.get("registry")
      spec = None
      if override_registry:
        spec = self._try_resolve_from_registry(
            override_registry, name, version, actual_repo_name
        )
      if not spec:
        spec = self._resolve_from_registries(name, version, actual_repo_name)
      if spec:
        self._apply_override_patches_to_spec(spec, override)
      return spec
    if override_type == "local_path":
      source_directory = pathlib.Path(override["path"])
      if not source_directory.is_absolute():
        source_directory = self._state.active_repo.source_directory.joinpath(
            override["path"]
        )
      return ResolvedModuleSpec(
          repository_id=RepositoryId(actual_repo_name),
          cmake_project_name=name,
          source_type="local_path",
          source_directory=source_directory,
          config={"cmake_name": name, "SOURCE_DIR": source_directory},
      )
    if override_type == "archive":
      config = {
          "cmake_name": name,
          "urls": override.get("urls"),
          "strip_prefix": override.get("strip_prefix"),
          "patch_cmds": override.get("patch_cmds"),
          "patches": override.get("patches"),
          "sha256": override.get("integrity"),
      }
      return ResolvedModuleSpec(
          repository_id=RepositoryId(actual_repo_name),
          cmake_project_name=name,
          source_type="archive",
          config=config,
      )
    if override_type == "git":
      config = {
          "cmake_name": name,
          "git_repository": override.get("remote"),
          "git_tag": override.get("commit") or override.get("tag"),
          "patch_cmds": override.get("patch_cmds"),
          "patches": override.get("patches"),
      }
      return ResolvedModuleSpec(
          repository_id=RepositoryId(actual_repo_name),
          cmake_project_name=name,
          source_type="git",
          config=config,
      )
    return None

  def _apply_module_override(
      self,
      name: str,
      version: str,
      actual_repo_name: str,
      override: dict[str, Any],
  ) -> None:
    if override.get("type") == "local_path":
      self._set_local_path_override(name, override["path"])
    spec = self._resolve_module_override(
        name, version, actual_repo_name, override
    )
    if spec:
      self._materialize_module_repo(spec)

  def _resolve_external_config(
      self, name: str, actual_repo_name: str, ext_config: dict[str, Any]
  ) -> ResolvedModuleSpec | None:
    if not ("urls" in ext_config or "url" in ext_config):
      return None
    repository_id = RepositoryId(actual_repo_name)
    config = ext_config.copy()
    cmake_name = config.get("cmake_project_name", name)
    config["cmake_name"] = cmake_name

    if repository_id in self._state.workspace.exclude_repositories:
      config.pop("bazel_to_cmake", None)
      config.pop("cmake_target_mapping", None)

    return ResolvedModuleSpec(
        repository_id=repository_id,
        cmake_project_name=cmake_name,
        source_type="archive",
        config=config,
    )

  def _apply_external_config(
      self, name: str, actual_repo_name: str, ext_config: dict[str, Any]
  ) -> None:
    spec = self._resolve_external_config(name, actual_repo_name, ext_config)
    if spec:
      self._materialize_module_repo(spec)

  def _resolve_from_registries(
      self, name: str, version: str, actual_repo_name: str
  ) -> ResolvedModuleSpec | None:
    """Resolves a module by searching through configured registries."""
    workspace_root = self._state.active_repo.source_directory
    registries = self._state.workspace._parsed_bazelrc.registries or [
        workspace_root.joinpath("third_party").as_posix()
    ]

    for registry in registries:
      registry = registry.replace("%workspace%", workspace_root.as_posix())
      spec = self._try_resolve_from_registry(
          registry, name, version, actual_repo_name
      )
      if spec:
        return spec
    return None

  def _try_resolve_from_registry(
      self, registry: str, name: str, version: str, actual_repo_name: str
  ) -> ResolvedModuleSpec | None:
    """Attempts to resolve a module from a specific registry."""
    # Only local registries are supported for now.
    if not registry.startswith("http"):
      if registry.startswith("file://"):
        registry = urllib.parse.unquote(urllib.parse.urlsplit(registry).path)
      registry_path = pathlib.Path(registry)
      module_dir = registry_path.joinpath("modules", name, version)
      module_config_path = module_dir.joinpath("bazel_to_cmake.json")
      source_json_path = module_dir.joinpath("source.json")

      if module_config_path.exists() or source_json_path.exists():
        return _resolve_spec_from_directory(
            repository_id=RepositoryId(actual_repo_name),
            directory=module_dir,
            registry_path=registry_path,
            workspace_root=self._state.active_repo.source_directory,
        )
    return None

  def _load_module_config(
      self,
      module_path: pathlib.Path,
      registry_path: pathlib.Path | None,
      name: str,
      version: str,
      actual_repo_name: str,
  ) -> CMakeRepository | None:
    spec = _resolve_spec_from_directory(
        repository_id=RepositoryId(actual_repo_name),
        directory=module_path,
        registry_path=registry_path,
        workspace_root=self._state.active_repo.source_directory,
    )
    return self._materialize_module_repo(spec)

  def _materialize_module_repo(
      self, spec: ResolvedModuleSpec
  ) -> CMakeRepository | None:
    """Materializes a resolved module spec into CMake repository and rules."""
    repository_id = spec.repository_id
    workspace = self._state.workspace
    repo = workspace.all_repositories.get(repository_id)

    fetch_content_base_dir = get_fetch_content_base_dir(self._state)
    source_dir = spec.source_directory
    binary_dir = None

    if spec.source_type == "local_path":
      if repo:
        source_dir = spec.source_directory or repo.source_directory
        binary_dir = repo.cmake_binary_dir
    else:
      if not source_dir or source_dir == pathlib.PurePath():
        if fetch_content_base_dir:
          source_dir = fetch_content_base_dir.joinpath(
              f"{spec.cmake_project_name.lower()}-src"
          )
      if fetch_content_base_dir:
        binary_dir = fetch_content_base_dir.joinpath(
            f"{spec.cmake_project_name.lower()}-build"
        )

    try:
      if not repo:
        repo = CMakeRepository.from_config(
            repository_id=repository_id,
            config=spec.config,
            source_directory=source_dir,
            cmake_binary_dir=binary_dir,
        )
        workspace.add_cmake_repository(repo)
        workspace.all_repositories[repository_id] = repo
      else:
        repo = repo.with_cmake_directories(
            source_directory=source_dir or repo.source_directory,
            cmake_binary_dir=binary_dir or repo.cmake_binary_dir,
        )
        workspace.all_repositories[repository_id] = repo

      cmake_project_name = str(repo.cmake_project_name)
      bazel_to_cmake_config = spec.config.get("bazel_to_cmake")
      if repo.source_directory and bazel_to_cmake_config is not None:
        new_cmakelists = io.StringIO()
        helpers.write_bazel_to_cmake_cmakelists(
            _context=self._state.evaluation_context,
            _new_cmakelists=new_cmakelists,
            _patch_commands=[],
            name=repository_id.repository_name,
            cmake_name=cmake_project_name,
            bazel_to_cmake=bazel_to_cmake_config,
            cmake_target_mapping=spec.config.get("cmake_target_mapping"),
            repo_mapping=spec.config.get("repo_mapping"),
            build_file=(
                spec.config.get("build_file")
                or bazel_to_cmake_config.get("build_file")
            ),
            cmake_extra_build_file=(
                spec.config.get("cmake_extra_build_file")
                or bazel_to_cmake_config.get("cmake_extra_build_file")
            ),
            is_local=True,
        )
        cmakelists_path = pathlib.Path(repo.source_directory).joinpath(
            "CMakeLists.txt"
        )
        try:
          cmakelists_path.parent.mkdir(parents=True, exist_ok=True)
          cmakelists_path.write_text(
              new_cmakelists.getvalue(), encoding="utf-8"
          )
          extra_bf = spec.config.get(
              "cmake_extra_build_file"
          ) or bazel_to_cmake_config.get("cmake_extra_build_file")
          if extra_bf:
            extra_bf_target = (
                self._state.evaluation_context.resolve_target_or_label(extra_bf)
            )
            extra_bf_path = self._state.evaluation_context.get_source_file_path(
                extra_bf_target
            )
            if extra_bf_path and pathlib.Path(extra_bf_path).exists():
              shutil.copy2(
                  extra_bf_path,
                  pathlib.Path(repo.source_directory) / "extraBUILD.bazel",
              )
          bf = spec.config.get("build_file") or bazel_to_cmake_config.get(
              "build_file"
          )
          if bf:
            bf_target = self._state.evaluation_context.resolve_target_or_label(
                bf
            )
            bf_path = self._state.evaluation_context.get_source_file_path(
                bf_target
            )
            if bf_path and pathlib.Path(bf_path).exists():
              shutil.copy2(
                  bf_path,
                  pathlib.Path(repo.source_directory) / "BUILD.bazel",
              )
          if spec.overlay_dir and spec.overlay_dir.exists():
            for overlay_file in spec.overlay_dir.glob("**/*"):
              if overlay_file.is_file():
                rel = overlay_file.relative_to(spec.overlay_dir)
                dst_f = pathlib.Path(repo.source_directory).joinpath(rel)
                dst_f.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(overlay_file, dst_f)
        except OSError as e:
          print(f"Warning: Failed to write {cmakelists_path}: {e}")

      fetch_kwargs = spec.config.copy()
      fetch_kwargs.setdefault("cmake_name", cmake_project_name)
      fetch_kwargs.setdefault("name", repository_id.repository_name)
      if repo.source_directory and repo.source_directory != pathlib.PurePath():
        fetch_kwargs.setdefault("SOURCE_DIR", repo.source_directory)

      third_party_http_archive._emit_fetch_content_impl(
          self._state.evaluation_context,
          repo,
          **fetch_kwargs,
      )
      return repo
    except Exception as e:
      print(f"Warning: Failed to process module {repository_id}: {e}")
      traceback.print_exc()
      return None

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
