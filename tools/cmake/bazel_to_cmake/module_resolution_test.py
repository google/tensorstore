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
"""Tests for module_resolution.py."""

import json
import pathlib
import unittest.mock

from .cmake_repository import CMakeRepository
from .cmake_target import CMakePackage
from .module_resolution import ModuleResolver
from .starlark.bazel_target import RepositoryId
from .workspace import Workspace


def test_set_module_name_version():
  state = unittest.mock.MagicMock()
  resolver = ModuleResolver(state)
  resolver.set_module_name_version("my_module", "1.0.0")
  assert resolver._module_name == "my_module"
  assert resolver._module_version == "1.0.0"


def test_load_lockfile():
  state = unittest.mock.MagicMock()
  resolver = ModuleResolver(state)
  
  lockfile_data = {
      "moduleExtensions": {
          "//:ext.bzl%ext": {
              "general": {
                  "generatedRepoSpecs": {
                      "repo": {
                          "repoRuleId": "http_archive",
                          "attributes": {
                              "urls": ["http://example.com"],
                          }
                      }
                  }
              }
          }
      }
  }
  
  resolver.load_lockfile(lockfile_data)
  
  assert "//:ext.bzl%ext" in resolver._lockfile_repos
  assert "repo" in resolver._lockfile_repos["//:ext.bzl%ext"]
  assert resolver._lockfile_repos["//:ext.bzl%ext"]["repo"]["repoRuleId"] == "http_archive"


def test_add_module_override_local_path():
  state = unittest.mock.MagicMock()
  state.active_repo.source_directory = pathlib.Path("/workspace")
  state.workspace.all_repositories = {}

  resolver = ModuleResolver(state)
  resolver.add_module_override(
      "my_dep", {"type": "local_path", "path": "third_party/my_dep"}
  )

  assert resolver._module_overrides["my_dep"] == {
      "type": "local_path",
      "path": "third_party/my_dep",
  }


def test_add_bazel_dep_with_override(tmp_path):
  state = unittest.mock.MagicMock()
  state.active_repo.source_directory = tmp_path
  repo_id = RepositoryId("my_dep")

  # Ensure the repo exists so _set_local_path_override is called by add_module_override
  existing_repo = CMakeRepository(
      repository_id=repo_id,
      cmake_project_name=CMakePackage("MyDep"),
      source_directory=pathlib.Path("src"),
      cmake_binary_dir=pathlib.Path("bin"),
      repo_mapping={},
      persisted_canonical_name={},
      executable_targets=set(),
  )
  state.workspace = unittest.mock.MagicMock(spec=Workspace)
  state.workspace.all_repositories = {repo_id: existing_repo}

  resolver = ModuleResolver(state)
  # This should call _set_local_path_override which calls add_cmake_repository
  resolver.add_module_override(
      "my_dep", {"type": "local_path", "path": "overridden_path"}
  )

  state.workspace.add_cmake_repository.assert_called_once()
  updated_repo = state.workspace.add_cmake_repository.call_args.args[0]
  assert updated_repo.source_directory == tmp_path.joinpath("overridden_path")


def test_resolve_from_registries(tmp_path):
  registry_path = tmp_path / "registry"
  module_path = registry_path / "modules" / "my_dep" / "1.0.0"
  module_path.mkdir(parents=True)

  config = {
      "cmake_project_name": "MyDep",
      "cmake_target_mapping": {"@my_dep//:my_target": "MyDep::MyTarget"},
  }
  (module_path / "bazel_to_cmake.json").write_text(json.dumps(config))

  source_config = {"type": "local_path", "path": "src"}
  (module_path / "source.json").write_text(json.dumps(source_config))

  state = unittest.mock.MagicMock()
  state.active_repo.source_directory = tmp_path

  # Use a real Workspace to get real behavior for add_cmake_repository
  ws = Workspace(
      RepositoryId("root"),
      {
          "FETCHCONTENT_BASE_DIR": (tmp_path / "deps").as_posix(),
          "CMAKE_CXX_COMPILER_ID": "GNU",
          "CMAKE_FIND_PACKAGE_REDIRECTS_DIR": (
              (tmp_path / "redirects").as_posix()
          ),
      },
  )
  ws._parsed_bazelrc.registries = [registry_path.as_posix()]
  state.workspace = ws
  state.evaluation_context.access.side_effect = (
      lambda t: state
      if t.__name__ == "EvaluationState"
      else unittest.mock.MagicMock()
  )

  # Ensure the redirects directory exists
  pathlib.Path(tmp_path / "redirects").mkdir(parents=True)

  resolver = ModuleResolver(state)
  with unittest.mock.patch("pathlib.Path.write_text"), unittest.mock.patch(
      "os.makedirs"
  ):
    resolver.add_bazel_dep("my_dep", "1.0.0", 0, "my_dep", False)

  # Verify repository was added to workspace
  assert RepositoryId("my_dep") in ws.all_repositories
  assert (
      ws.all_repositories[RepositoryId("my_dep")].cmake_project_name == "MyDep"
  )


def test_include_module_file(tmp_path):
  include_file = tmp_path / "extra.MODULE.bazel"
  include_file.write_text("print('hello from included file')")

  state = unittest.mock.MagicMock()
  target_id = unittest.mock.MagicMock()
  target_id.repository_id = RepositoryId("")
  target_id.package_name = ""
  target_id.target_name = "extra.MODULE.bazel"

  state.evaluation_context.resolve_target_or_label.return_value = target_id
  state.evaluation_context.workspace_root_for_label.return_value = pathlib.Path(
      tmp_path
  ).as_posix()

  resolver = ModuleResolver(state)

  # Use the ModuleResolver's module for patching to avoid ModuleNotFoundError
  with unittest.mock.patch(
      f"{ModuleResolver.__module__}.compile_and_exec"
  ) as mock_exec:
    resolver.include_module_file("//:extra.MODULE.bazel", {})
    mock_exec.assert_called_once()
    assert mock_exec.call_args.args[0] == "print('hello from included file')"


def test_resolve_from_registry_no_json(tmp_path):
  registry_path = tmp_path / "registry"
  module_path = registry_path / "modules" / "my_dep" / "1.0.0"
  module_path.mkdir(parents=True)

  # No bazel_to_cmake.json created here

  source_config = {"type": "local_path", "path": "src"}
  (module_path / "source.json").write_text(json.dumps(source_config))

  state = unittest.mock.MagicMock()
  state.active_repo.source_directory = tmp_path

  ws = Workspace(
      RepositoryId("root"),
      {
          "FETCHCONTENT_BASE_DIR": (tmp_path / "deps").as_posix(),
          "CMAKE_CXX_COMPILER_ID": "GNU",
          "CMAKE_FIND_PACKAGE_REDIRECTS_DIR": (
              (tmp_path / "redirects").as_posix()
          ),
      },
  )
  ws._parsed_bazelrc.registries = [registry_path.as_posix()]
  state.workspace = ws

  resolver = ModuleResolver(state)
  with unittest.mock.patch(
      f"{ModuleResolver.__module__}.third_party_http_archive._emit_fetch_content_impl"
  ) as mock_emit:
    resolver.add_bazel_dep("my_dep", "1.0.0", 0, "my_dep", False)

    mock_emit.assert_called_once()
    # bazel_to_cmake should be None because there's no config file
    assert mock_emit.call_args.kwargs.get("bazel_to_cmake") is None


def test_resolve_from_registry_no_bazel_to_cmake_key(tmp_path):
  registry_path = tmp_path / "registry"
  module_path = registry_path / "modules" / "my_dep" / "1.0.0"
  module_path.mkdir(parents=True)

  # bazel_to_cmake.json exists but lacks "bazel_to_cmake" key
  config = {"cmake_project_name": "MyDep"}
  (module_path / "bazel_to_cmake.json").write_text(json.dumps(config))

  source_config = {"type": "local_path", "path": "src"}
  (module_path / "source.json").write_text(json.dumps(source_config))

  state = unittest.mock.MagicMock()
  state.active_repo.source_directory = tmp_path

  ws = Workspace(
      RepositoryId("root"),
      {
          "FETCHCONTENT_BASE_DIR": (tmp_path / "deps").as_posix(),
          "CMAKE_CXX_COMPILER_ID": "GNU",
          "CMAKE_FIND_PACKAGE_REDIRECTS_DIR": (
              (tmp_path / "redirects").as_posix()
          ),
      },
  )
  ws._parsed_bazelrc.registries = [registry_path.as_posix()]
  state.workspace = ws

  resolver = ModuleResolver(state)
  with unittest.mock.patch(
      f"{ModuleResolver.__module__}.third_party_http_archive._emit_fetch_content_impl"
  ) as mock_emit:
    resolver.add_bazel_dep("my_dep", "1.0.0", 0, "my_dep", False)

    mock_emit.assert_called_once()
    # bazel_to_cmake should be None because the key is missing in config
    assert mock_emit.call_args.kwargs.get("bazel_to_cmake") is None


def test_add_module_override_archive():
  state = unittest.mock.MagicMock()
  resolver = ModuleResolver(state)
  override_info = {"type": "archive", "urls": ["http://example.com"]}
  resolver.add_module_override("my_module", override_info)
  assert resolver._module_overrides["my_module"] == override_info


def test_add_module_override_git():
  state = unittest.mock.MagicMock()
  resolver = ModuleResolver(state)
  override_info = {"type": "git", "remote": "http://github.com"}
  resolver.add_module_override("my_module", override_info)
  assert resolver._module_overrides["my_module"] == override_info


def test_apply_module_override_archive():
  state = unittest.mock.MagicMock()
  state.workspace.all_repositories = {}
  resolver = ModuleResolver(state)
  override = {
      "type": "archive",
      "urls": ["http://example.com"],
      "integrity": "sha256-...",
  }
  with unittest.mock.patch(
      f"{ModuleResolver.__module__}.third_party_http_archive._emit_fetch_content_impl"
  ) as mock_emit:
    resolver._apply_module_override("my_dep", "1.0.0", "my_dep", override)
    mock_emit.assert_called_once()


def test_apply_module_override_git():
  state = unittest.mock.MagicMock()
  state.workspace.all_repositories = {}
  resolver = ModuleResolver(state)
  override = {
      "type": "git",
      "remote": "http://github.com/repo",
      "commit": "123456",
  }
  with unittest.mock.patch(
      f"{ModuleResolver.__module__}.third_party_http_archive._emit_fetch_content_impl"
  ) as mock_emit:
    resolver._apply_module_override("my_dep", "1.0.0", "my_dep", override)
    mock_emit.assert_called_once()


def test_use_repo():
  state = unittest.mock.MagicMock()
  state.workspace.all_repositories = {}
  resolver = ModuleResolver(state)
  
  lockfile_data = {
      "moduleExtensions": {
          "//:ext.bzl%ext": {
              "general": {
                  "generatedRepoSpecs": {
                      "remote_repo": {
                          "repoRuleId": "http_archive",
                          "attributes": {
                              "urls": ["http://example.com"],
                          }
                      }
                  }
              }
          }
      }
  }
  
  resolver.load_lockfile(lockfile_data)
  
  proxy = unittest.mock.MagicMock()
  proxy.bzl_file = "//:ext.bzl"
  proxy.name = "ext"
  
  with unittest.mock.patch(
      f"{ModuleResolver.__module__}.third_party_http_archive._emit_fetch_content_impl"
  ) as mock_emit:
    resolver.use_repo(proxy, local_name="remote_repo")
    
    state.workspace.add_cmake_repository.assert_called_once()
    repo = state.workspace.add_cmake_repository.call_args.args[0]
    assert repo.repository_id == RepositoryId("local_name")
    mock_emit.assert_called_once()


