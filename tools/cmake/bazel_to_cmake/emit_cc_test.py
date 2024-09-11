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
"""Tests for emit_cc functions."""

# pylint: disable=g-importing-member
import pathlib

from .cmake_repository import CMakeRepository
from .cmake_target import CMakePackage
from .emit_cc import construct_cc_includes
from .emit_cc import replace_with_cmake_macro_dirs
from .starlark.bazel_target import PackageId
from .starlark.bazel_target import RepositoryId

REPO = CMakeRepository(
    RepositoryId("foo"),
    CMakePackage("Foo"),
    pathlib.PurePath("foo-srcdir"),
    pathlib.PurePath("foo-bindir"),
    {},
    {},
)


def test_replace_with_cmake_macro_dirs():
  assert [
      "${PROJECT_BINARY_DIR}/b",
      "${PROJECT_SOURCE_DIR}/c",
      "a",
  ] == replace_with_cmake_macro_dirs(
      REPO,
      [
          "a",
          REPO.cmake_binary_dir.joinpath("b"),
          REPO.source_directory.joinpath("c"),
      ],
  )


def test_construct_cc_includes_bare():
  # No includes
  assert not construct_cc_includes(
      REPO,
      PackageId("foo", "bar"),
  )
  # Individual srcdir / bindir includes
  assert ["${PROJECT_SOURCE_DIR}"] == construct_cc_includes(
      REPO,
      PackageId("foo", "bar"),
      known_include_files=["foo-srcdir/bar/a.inc"],
  )
  assert ["${PROJECT_BINARY_DIR}"] == construct_cc_includes(
      REPO,
      PackageId("foo", "bar"),
      known_include_files=["foo-bindir/bar/b.inc"],
  )
  # Other combinations
  assert [
      "${PROJECT_BINARY_DIR}",
      "${PROJECT_SOURCE_DIR}",
  ] == construct_cc_includes(
      REPO,
      PackageId("foo", "bar"),
      known_include_files=[
          "foo-srcdir/bar/e.inc",
          "foo-bindir/bar/ee.h",
      ],
  )


def test_construct_cc_includes_includes():
  # includes does not test for file presence.
  assert [
      "${PROJECT_BINARY_DIR}/bar",
      "${PROJECT_SOURCE_DIR}/bar",
  ] == construct_cc_includes(
      REPO,
      PackageId("foo", "bar"),
      includes=["."],
  )
  # package-relative
  assert [
      "${PROJECT_BINARY_DIR}/bar/includes",
      "${PROJECT_SOURCE_DIR}/bar/includes",
  ] == construct_cc_includes(
      REPO,
      PackageId("foo", "bar"),
      includes=["includes"],
      known_include_files=["foo-srcdir/includes/b.h"],
  )
  assert [
      "${PROJECT_BINARY_DIR}/bar/includes",
      "${PROJECT_SOURCE_DIR}/bar/includes",
  ] == construct_cc_includes(
      REPO,
      PackageId("foo", "bar"),
      includes=["includes"],
      known_include_files=["foo-srcdir/bar/includes/c.h"],
  )
  # reposnitory-relative
  # bazel doesn't generate this one; it probably builds file symlinks.
  assert [
      "${PROJECT_BINARY_DIR}/includes",
      "${PROJECT_SOURCE_DIR}/includes",
  ] == construct_cc_includes(
      REPO,
      PackageId("foo", "bar"),
      includes=["/includes"],
      known_include_files=["foo-srcdir/includes/a.h"],
  )


def test_construct_cc_includes_include_prefix():
  # include_prefix really doesn't work for bazel_to_cmake.
  assert [
      "${PROJECT_SOURCE_DIR}",
  ] == construct_cc_includes(
      REPO,
      PackageId("foo", "bar"),
      include_prefix="bar",
      known_include_files=["foo-srcdir/bar/x.h"],
  )
  assert [
      "${PROJECT_SOURCE_DIR}",
  ] == construct_cc_includes(
      REPO,
      PackageId("foo", "bar"),
      include_prefix="_mismatch_",
      known_include_files=["foo-srcdir/bar/y.h"],
  )


def test_construct_cc_includes_strip_include_prefix():
  # mismatched
  assert [
      "${PROJECT_SOURCE_DIR}",
  ] == construct_cc_includes(
      REPO,
      PackageId("foo", "bar"),
      strip_include_prefix="xyz",
      known_include_files=["foo-srcdir/bar/a.h"],
  )
  assert [
      "${PROJECT_BINARY_DIR}",
  ] == construct_cc_includes(
      repo=REPO,
      strip_include_prefix="xyz",
      current_package_id=PackageId("foo", "bar"),
      known_include_files=["foo-bindir/bar/b.h"],
  )
  # Respoitory relative
  assert [
      "${PROJECT_SOURCE_DIR}/bar",
  ] == construct_cc_includes(
      REPO,
      PackageId("foo", "bar"),
      strip_include_prefix="/bar",
      known_include_files=["foo-srcdir/bar/c.h"],
  )
  assert [
      "${PROJECT_BINARY_DIR}/bar",
      "${PROJECT_SOURCE_DIR}/bar",
  ] == construct_cc_includes(
      REPO,
      PackageId("foo", "bar"),
      strip_include_prefix="/bar",
      known_include_files=["foo-srcdir/bar/d.h", "foo-bindir/bar/dd.h"],
  )
  assert [
      "${PROJECT_SOURCE_DIR}/bar",
  ] == construct_cc_includes(
      REPO,
      PackageId("foo", "bar"),
      strip_include_prefix="/bar",
      known_include_files=["foo-srcdir/bar/e.inc", "foo-srcdir/bar/e.h"],
  )
  # Package includes
  assert [
      "${PROJECT_SOURCE_DIR}",
  ] == construct_cc_includes(
      repo=REPO,
      strip_include_prefix="bar",
      current_package_id=PackageId("foo", "bar"),
      known_include_files=["foo-srcdir/bar/f.h"],
  )
  assert [
      "${PROJECT_SOURCE_DIR}/bar/includes",
  ] == construct_cc_includes(
      REPO,
      PackageId("foo", "bar"),
      strip_include_prefix="includes",
      known_include_files=["foo-srcdir/bar/includes/g.h"],
  )
