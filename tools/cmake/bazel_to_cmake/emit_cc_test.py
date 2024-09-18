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

from .emit_cc import construct_cc_includes
from .starlark.bazel_target import PackageId


SRCDIR = pathlib.PurePath("foo-srcdir")
BINDIR = pathlib.PurePath("foo-bindir")


def test_construct_cc_includes_bare():
  # No includes
  assert not construct_cc_includes(
      PackageId("foo", "bar"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
  )
  # Individual srcdir / bindir includes
  assert set([SRCDIR]) == construct_cc_includes(
      PackageId("foo", "bar"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      known_include_files=["foo-srcdir/bar/a.inc"],
  )
  assert set([BINDIR]) == construct_cc_includes(
      PackageId("foo", "bar"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      known_include_files=["foo-bindir/bar/b.inc"],
  )
  # Other combinations
  assert set([
      BINDIR,
      SRCDIR,
  ]) == construct_cc_includes(
      PackageId("foo", "bar"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      known_include_files=[
          "foo-srcdir/bar/e.inc",
          "foo-bindir/bar/ee.h",
      ],
  )


def test_construct_cc_includes_includes():
  # includes does not test for file presence.
  assert set([
      BINDIR.joinpath("bar"),
      SRCDIR.joinpath("bar"),
  ]) == construct_cc_includes(
      PackageId("foo", "bar"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      includes=["."],
  )
  # package-relative
  assert set([
      BINDIR.joinpath("bar/includes"),
      SRCDIR.joinpath("bar/includes"),
  ]) == construct_cc_includes(
      PackageId("foo", "bar"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      includes=["includes"],
      known_include_files=["foo-srcdir/includes/b.h"],
  )
  assert set([
      BINDIR.joinpath("bar/includes"),
      SRCDIR.joinpath("bar/includes"),
  ]) == construct_cc_includes(
      PackageId("foo", "bar"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      includes=["includes"],
      known_include_files=["foo-srcdir/bar/includes/c.h"],
  )
  # reposnitory-relative
  # bazel doesn't generate this one; it probably builds file symlinks.
  assert set([
      BINDIR.joinpath("includes"),
      SRCDIR.joinpath("includes"),
  ]) == construct_cc_includes(
      PackageId("foo", "bar"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      includes=["/includes"],
      known_include_files=["foo-srcdir/includes/a.h"],
  )


def test_construct_cc_includes_include_prefix():
  # include_prefix really doesn't work for bazel_to_cmake.
  assert set([
      SRCDIR,
  ]) == construct_cc_includes(
      PackageId("foo", "bar"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      include_prefix="bar",
      known_include_files=["foo-srcdir/bar/x.h"],
  )
  assert set([
      SRCDIR,
  ]) == construct_cc_includes(
      PackageId("foo", "bar"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      include_prefix="_mismatch_",
      known_include_files=["foo-srcdir/bar/y.h"],
  )


def test_construct_cc_includes_strip_include_prefix():
  # mismatched
  assert set([
      SRCDIR,
  ]) == construct_cc_includes(
      PackageId("foo", "bar"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      strip_include_prefix="xyz",
      known_include_files=["foo-srcdir/bar/a.h"],
  )
  assert set([
      BINDIR,
  ]) == construct_cc_includes(
      PackageId("foo", "bar"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      strip_include_prefix="xyz",
      known_include_files=["foo-bindir/bar/b.h"],
  )
  # Respoitory relative
  assert set([
      SRCDIR.joinpath("bar"),
  ]) == construct_cc_includes(
      PackageId("foo", "bar"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      strip_include_prefix="/bar",
      known_include_files=["foo-srcdir/bar/c.h"],
  )
  assert set([
      BINDIR.joinpath("bar"),
      SRCDIR.joinpath("bar"),
  ]) == construct_cc_includes(
      PackageId("foo", "bar"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      strip_include_prefix="/bar",
      known_include_files=["foo-srcdir/bar/d.h", "foo-bindir/bar/dd.h"],
  )
  assert set([
      SRCDIR.joinpath("bar"),
  ]) == construct_cc_includes(
      PackageId("foo", "bar"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      strip_include_prefix="/bar",
      known_include_files=["foo-srcdir/bar/e.inc", "foo-srcdir/bar/e.h"],
  )
  # Package includes
  assert set([
      SRCDIR,
  ]) == construct_cc_includes(
      PackageId("foo", "bar"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      strip_include_prefix="bar",
      known_include_files=["foo-srcdir/bar/f.h"],
  )
  assert set([
      SRCDIR.joinpath("bar/includes"),
  ]) == construct_cc_includes(
      PackageId("foo", "bar"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      strip_include_prefix="includes",
      known_include_files=["foo-srcdir/bar/includes/g.h"],
  )
