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
"""Tests for utility functions."""

# pylint: disable=relative-beyond-top-level

import os
import pathlib
from typing import Any, List, Optional

import pytest

from .cmake_target import CMakeExecutableTargetProvider
from .cmake_target import CMakePackage
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetPair
from .starlark.bazel_target import PackageId
from .starlark.bazel_target import RepositoryId
from .starlark.bazel_target import TargetId
from .starlark.common_providers import FilesProvider
from .starlark.invocation_context import InvocationContext
from .starlark.provider import Provider
from .starlark.provider import TargetInfo
from .starlark.toolchain import CMAKE_TOOLCHAIN
from .variable_substitution import apply_location_and_make_variable_substitutions
from .variable_substitution import apply_location_substitutions
from .variable_substitution import apply_make_variable_substitutions
from .variable_substitution import do_bash_command_replacement
from .variable_substitution import generate_substitutions


class MyContext(InvocationContext):

  def __init__(self):
    self._caller_package_id = PackageId.parse("@foo//bar/baz")
    self._relative_to = os.path.dirname(__file__)

  @property
  def relative_to(self) -> str:
    return self._relative_to

  @property
  def caller_package_id(self) -> PackageId:
    return self._caller_package_id

  def access(self, provider_type: Any) -> "MyContext":
    del provider_type
    return self

  def apply_repo_mapping(
      self, target: TargetId, mapping_repository_id: Optional[RepositoryId]
  ) -> TargetId:
    return target

  def resolve_source_root(
      self, repository_id: RepositoryId
  ) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath(
        f"{self.relative_to}/external/{repository_id.repository_name}"
    )

  def resolve_output_root(
      self, repository_id: RepositoryId
  ) -> pathlib.PurePath:
    return pathlib.PurePosixPath(
        f"{self.relative_to}/bindir/{repository_id.repository_name}"
    )

  def get_optional_target_info(
      self, target_id: TargetId
  ) -> Optional[TargetInfo]:
    providers: List[Provider] = []
    if "self" in repr(target_id):
      providers.append(FilesProvider([__file__]))
    if "file" in repr(target_id):
      providers.append(
          FilesProvider([f"{target_id.package_name}/{target_id.target_name}"])
      )
    if "cmake" in repr(target_id):
      providers.append(
          CMakeExecutableTargetProvider(CMakeTarget(target_id.target_name))
      )
    if not providers:
      return None
    return TargetInfo(*providers)

  def get_target_info(self, target_id: TargetId) -> TargetInfo:
    info = self.get_optional_target_info(target_id)
    assert info is not None
    return info

  def get_dep(
      self, target_id: TargetId, alias: bool = True
  ) -> List[CMakeTarget]:
    del alias
    t0 = f"cmake_{target_id.target_name}".replace("/", "_")
    if "none" in t0:
      return []
    else:
      return [CMakeTarget(t0)]

  def generate_cmake_target_pair(self, target_id: TargetId, alias: bool = True):
    return CMakeTargetPair(
        CMakePackage("cmake_project"),
        CMakeTarget(f"cmake_project_{target_id.target_name}"),
        CMakeTarget(f"cmake_project::{target_id.target_name}"),
    )


def test_apply_make_variable_substitutions():
  ctx = MyContext()

  with pytest.raises(Exception) as _:
    apply_make_variable_substitutions(ctx, "$$(location $(foo)) $@", {}, [])

  subs = {"foo": "bar", "@": "x", "<": "y"}

  assert "bar x" == apply_make_variable_substitutions(
      ctx, "$(foo) $@", subs, []
  )

  assert "$(location bar) x" == apply_make_variable_substitutions(
      ctx, "$$(location $(foo)) $@", subs, []
  )

  assert "ls $(dirname $x)" == apply_make_variable_substitutions(
      ctx, "ls $$(dirname $$x)", subs, []
  )

  assert "${bar} x" == apply_make_variable_substitutions(
      ctx, "$${bar} $@", subs, []
  )

  assert "${CMAKE_COMMAND} x" == apply_make_variable_substitutions(
      ctx,
      "$(CMAKE_COMMAND) $@",
      subs,
      toolchains=[TargetId.parse(CMAKE_TOOLCHAIN)],
  )


def test_apply_location_substitutions():
  ctx = MyContext()

  assert "$(frob my/file/a.h) $@" == apply_location_substitutions(
      ctx, "$(frob my/file/a.h) $@", ""
  )

  assert "variable_substitution_test.py $@" == apply_location_substitutions(
      ctx, "$(location my/self/a.h) $@", ctx.relative_to
  )

  assert "$<TARGET_FILE:cmake/target> $@" == apply_location_substitutions(
      ctx, "$(location cmake/target) $@", ctx.relative_to
  )

  assert "$<TARGET_FILE:cmake_project_foo>" == apply_location_substitutions(
      ctx, "$(location @other//foo)", "", []
  )

  # TargetInfo with no targets and no CMakeTarget name raises an exception
  with pytest.raises(Exception):
    apply_location_substitutions(ctx, "$(location a/none/target)", "", [])


def test_apply_location_and_make_variable_substitutions():
  ctx = MyContext()

  subs = {"foo": "some/file.h", "@": "x", "<": "y"}

  assert (
      "ls $(dirname bar/baz/some/file.h)"
      == apply_location_and_make_variable_substitutions(
          ctx,
          cmd="ls $$(dirname $(location $(foo)))",
          relative_to="",
          custom_target_deps=[],
          substitutions=subs,
          toolchains=None,
      )
  )


def test_generate_substitutions():
  ctx = MyContext()
  a_h = (
      ctx.get_source_file_path(ctx.caller_package_id.parse_target("file/a.h"))
      or pathlib.PurePosixPath("__missing__")
  ).as_posix()
  b_h = (
      ctx.get_source_file_path(ctx.caller_package_id.parse_target("file/b.h"))
      or pathlib.PurePosixPath("__missing__")
  ).as_posix()
  a_cc = ctx.get_generated_file_path(
      ctx.caller_package_id.parse_target("file/a.cc")
  )
  b_cc = ctx.get_generated_file_path(
      ctx.caller_package_id.parse_target("file/b.cc")
  )

  assert generate_substitutions(
      ctx,
      ctx.caller_package_id.parse_target("file"),
      src_files=[a_h],
      out_files=[a_cc],
  ) == {
      "<": '"bar/baz/file/a.h"',
      "@": f'"{ctx.relative_to}/bindir/foo/bar/baz/file/a.cc"',
      "@D": f'"{ctx.relative_to}/bindir/foo/bar/baz/file"',
      "BINDIR": f"{ctx.relative_to}/bindir/foo",
      "GENDIR": f"{ctx.relative_to}/bindir/foo",
      "OUTS": f'"{ctx.relative_to}/bindir/foo/bar/baz/file/a.cc"',
      "RULEDIR": f"{ctx.relative_to}/bindir/foo/bar/baz",
      "SRCS": '"bar/baz/file/a.h"',
  }

  assert generate_substitutions(
      ctx,
      ctx.caller_package_id.parse_target("file"),
      src_files=[a_h, b_h],
      out_files=[a_cc, b_cc],
  ) == {
      "@D": f"{ctx.relative_to}/bindir/foo/bar/baz",
      "BINDIR": f"{ctx.relative_to}/bindir/foo",
      "GENDIR": f"{ctx.relative_to}/bindir/foo",
      "OUTS": (
          f'"{ctx.relative_to}/bindir/foo/bar/baz/file/a.cc"'
          f' "{ctx.relative_to}/bindir/foo/bar/baz/file/b.cc"'
      ),
      "RULEDIR": ctx.relative_to + "/bindir/foo/bar/baz",
      "SRCS": '"bar/baz/file/a.h" "bar/baz/file/b.h"',
  }


def test_do_bash_command_replacement():
  assert "ls bar/baz/some" == do_bash_command_replacement(
      "ls $(dirname bar/baz/some/file.h)"
  )

  assert "ls bar/baz/some/.." == do_bash_command_replacement(
      "ls $(dirname bar/baz/some/file.h)/.."
  )

  assert "bar/baz/some" == do_bash_command_replacement(
      "$(dirname bar/baz/some/..)"
  )

  assert "$<TARGET_FILE:gen> bar/baz/some" == do_bash_command_replacement(
      "$<TARGET_FILE:gen> $(dirname bar/baz/some/file.h)"
  )

  assert "bar/baz" == do_bash_command_replacement(
      "$(dirname $(dirname bar/baz/some/file.h))"
  )

  assert "ls bar/baz" == do_bash_command_replacement(
      "ls $(dirname\n$(dirname\nbar/baz/some/file.h))"
  )

  assert "./a" == do_bash_command_replacement("$(dirname $(dirname a)/a/b)")

  assert "." == do_bash_command_replacement('$(dirname $(dirname "x.h" ))')

  with pytest.raises(Exception):
    do_bash_command_replacement("$(dirname )")
