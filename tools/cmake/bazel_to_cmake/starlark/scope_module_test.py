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

# pylint: disable=g-importing-member,unused-argument

import pathlib
import tempfile
from typing import Any, cast
from unittest import mock

from ..active_repository import Repository
from ..evaluation_impl import EvaluationImpl
from ..evaluation_state import EvaluationState
from ..workspace import Workspace
from .bazel_target import PackageId
from .bazel_target import RepositoryId
from .bazel_target import TargetId
from .exec import compile_and_exec
from .ignored import IgnoredLibrary
from .invocation_context import InvocationContext
from .invocation_context import RuleImpl
from .scope_module_file import ScopeModuleFile
from .struct import Struct


MODULE_GLOBALS = """
def test_all():
    archive_override(module_name="foo", urls=["http://foo"])
    bazel_dep(name="bar", version="1.0")
    git_override(module_name="baz", remote="http://baz")
    include("//:foo.bazel")
    inject_repo(struct(), "repo")
    local_path_override(module_name="qux", path="qux")
    module(name="my_module", version="1.0")
    multiple_version_override(module_name="multi", versions=["1.0", "2.0"])
    register_execution_platforms("//:plat")
    register_toolchains("//:tool")
    single_version_override(module_name="single", version="1.0")
    ext = use_extension("//:ext.bzl", "ext")
    use_repo(ext, "repo")
    use_repo_rule("//:rule.bzl", "my_rule")
"""


class MyContext(InvocationContext):

  def __init__(self):
    self._caller_package_id = PackageId.parse("@foo//bar/baz")
    self.workspace = Workspace(
        RepositoryId("foo"), {"PROJECT_IS_TOP_LEVEL": "TRUE"}
    )
    # Mock the CMakeRepository inside all_repositories
    mock_cmake_repo = type(
        "MockCMakeRepo",
        (),
        {
            "cmake_project_name": "foo_project",
            "source_directory": pathlib.Path("/"),
            "cmake_binary_dir": pathlib.Path("/bin"),
            "repo_mapping": {},
            "repository_id": RepositoryId("foo"),
        },
    )
    self.workspace.all_repositories[RepositoryId("foo")] = mock_cmake_repo

    mock_repo = Repository(self.workspace, RepositoryId("foo"), {}, True)
    self.evaluation_state = EvaluationImpl(mock_repo)
    self._workspace_root = pathlib.Path("/")
    self.module_name = None
    self.module_version = None
    self.overrides = {}
    self.deps = []
    self.use_repo_calls = []

  def use_repo(self, extension_proxy, *args, **kwargs):
    self.use_repo_calls.append((extension_proxy, args, kwargs))

  @property
  def caller_package_id(self):
    return self._caller_package_id

  def apply_repo_mapping(
      self, target: TargetId, mapping_repository_id: RepositoryId | None
  ) -> TargetId:
    return target

  def resolve_source_root(
      self, repository_id: RepositoryId
  ) -> pathlib.PurePath:
    return pathlib.PurePath(f"external/{repository_id.repository_name}")

  def load_library(self, target: TargetId) -> dict[str, Any]:
    return IgnoredLibrary()

  def add_rule(
      self,
      rule_id: TargetId,
      impl: RuleImpl,
      outs: list[TargetId] | None = None,
      **kwargs,
  ) -> None:
    pass

  def access(self, provider_type: type[Any]):
    if provider_type == EvaluationState:
      return self.evaluation_state
    return super().access(provider_type)

  def workspace_root_for_label(self, repository_id: RepositoryId) -> str:
    return self._workspace_root.as_posix()

  def set_module_info(self, name, version, compatibility_level, repo_name):
    self.module_name = name
    self.module_version = version

  def add_bazel_dep(
      self, name, version, max_compatibility_level, repo_name, dev_dependency
  ):
    self.deps.append({"name": name, "version": version})

  def add_module_override(self, module_name, override_info):
    self.overrides[module_name] = override_info

  def include_module_file(self, label, scope=None):
    if scope is None:
      return
    target_id = self.resolve_target_or_label(label)
    path = pathlib.Path(
        self.workspace_root_for_label(target_id.repository_id)
    ).joinpath(target_id.package_name, target_id.target_name)
    with open(path, "r", encoding="utf-8") as f:
      content = f.read()
    compile_and_exec(content, str(path), scope)


def test_module_globals():
  with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = pathlib.Path(tmpdir)
    (tmp_path / "foo.bazel").write_text("", encoding="utf-8")

    ctx = MyContext()
    ctx._workspace_root = tmp_path

    scope = ScopeModuleFile(
        ctx, TargetId.parse("@foo//:MODULE.bazel"), "foo/MODULE.bazel"
    )

    compile_and_exec(MODULE_GLOBALS, "foo/MODULE.bazel", scope)
    # If it doesn't raise KeyError, then all globals are present.
    scope["test_all"]()


def test_module_overrides():
  scope = ScopeModuleFile(
      MyContext(), TargetId.parse("@foo//:MODULE.bazel"), "foo/MODULE.bazel"
  )

  scope["local_path_override"](module_name="my_mod", path="/path/to/mod")
  ctx = cast(MyContext, scope._context)
  assert ctx.overrides["my_mod"]["type"] == "local_path"
  assert ctx.overrides["my_mod"]["path"] == "/path/to/mod"

  scope["archive_override"](
      module_name="arch", urls=["http://url"], integrity="abc"
  )
  assert ctx.overrides["arch"]["type"] == "archive"
  assert ctx.overrides["arch"]["urls"] == ["http://url"]
  assert ctx.overrides["arch"]["integrity"] == "abc"

  scope["git_override"](module_name="gt", remote="http://git", commit="123")
  assert ctx.overrides["gt"]["type"] == "git"
  assert ctx.overrides["gt"]["remote"] == "http://git"


def test_bazel_module():
  scope = ScopeModuleFile(
      MyContext(), TargetId.parse("@foo//:MODULE.bazel"), "foo/MODULE.bazel"
  )
  scope["module"](name="test_name", version="1.2.3")
  ctx = cast(MyContext, scope._context)
  assert ctx.module_name == "test_name"
  assert ctx.module_version == "1.2.3"


def test_module_include():
  with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = pathlib.Path(tmpdir)
    include_file = tmp_path / "extra.bazel"
    include_file.write_text(
        "bazel_dep(name='bar', version='1.0')", encoding="utf-8"
    )

    ctx = MyContext()
    # Override workspace_root_for_label to point to our temp dir
    ctx._workspace_root = tmp_path

    scope = ScopeModuleFile(
        ctx, TargetId.parse("@foo//:MODULE.bazel"), "foo/MODULE.bazel"
    )
    # We can use the real include_module_file or a mock.
    # Here we use the one implemented in MyContext.

    scope.bazel_include("//:extra.bazel")
    assert ctx.deps[0]["name"] == "bar"


def test_bazel_use_repo():
  ctx = MyContext()
  scope = ScopeModuleFile(
      ctx, TargetId.parse("@foo//:MODULE.bazel"), "foo/MODULE.bazel"
  )
  proxy = Struct(bzl_file="//:ext.bzl", name="ext")
  scope["use_repo"](proxy, "repo1", repo2_local="repo2_remote")

  assert len(ctx.use_repo_calls) == 1
  call = ctx.use_repo_calls[0]
  assert call[0] == proxy
  assert call[1] == ("repo1",)
  assert call[2] == {"repo2_local": "repo2_remote"}


def test_bazel_use_extension():
  ctx = MyContext()
  scope = ScopeModuleFile(
      ctx, TargetId.parse("@foo//:MODULE.bazel"), "foo/MODULE.bazel"
  )

  # Mock load_library to return a dummy extension
  mock_library = {"my_ext": Struct(implementation=lambda ctx: None)}
  ctx.load_library = mock.MagicMock(return_value=mock_library)
  ctx.resolve_target_or_label = mock.MagicMock(
      return_value=TargetId.parse("@foo//:ext.bzl")
  )

  proxy = scope["use_extension"]("//:ext.bzl", "my_ext")

  assert proxy.bzl_file == "//:ext.bzl"
  assert proxy.name == "my_ext"
