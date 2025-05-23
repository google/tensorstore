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
  target_includes = construct_cc_includes(
      PackageId("foo", "aaa"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
  )
  assert not target_includes.system
  assert not target_includes.public
  assert set(["${PROJECT_SOURCE_DIR}"]) == target_includes.private

  # Individual srcdir / bindir includes
  target_includes = construct_cc_includes(
      PackageId("foo", "bbb"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      known_include_files=["foo-srcdir/bbb/a.inc"],
  )
  assert set([SRCDIR]) == target_includes.public
  assert set(["${PROJECT_SOURCE_DIR}"]) == target_includes.private

  target_includes = construct_cc_includes(
      PackageId("foo", "ccc"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      known_include_files=["foo-bindir/ccc/b.inc"],
  )
  assert set([BINDIR]) == target_includes.public
  assert (
      set(["${PROJECT_BINARY_DIR}", "${PROJECT_SOURCE_DIR}"])
      == target_includes.private
  )

  # Other combinations
  target_includes = construct_cc_includes(
      PackageId("foo", "ddd"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      known_include_files=[
          "foo-srcdir/ddd/e.inc",
          "foo-bindir/ddd/ee.h",
      ],
  )
  assert (
      set([
          BINDIR,
          SRCDIR,
      ])
      == target_includes.public
  )


def test_construct_cc_includes_includes():
  # includes does not test for file presence.
  target_includes = construct_cc_includes(
      PackageId("foo", "eee"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      includes=["."],
  )
  assert (
      set([
          BINDIR.joinpath("eee"),
          SRCDIR.joinpath("eee"),
      ])
      == target_includes.system
  )

  # package-relative
  target_includes = construct_cc_includes(
      PackageId("foo", "fff"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      includes=["includes"],
      known_include_files=["foo-srcdir/includes/b.h"],
  )
  assert (
      set([
          BINDIR.joinpath("fff/includes"),
          SRCDIR.joinpath("fff/includes"),
      ])
      == target_includes.system
  )
  assert not target_includes.public

  target_includes = construct_cc_includes(
      PackageId("foo", "ggg"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      includes=["includes"],
      known_include_files=["foo-srcdir/ggg/includes/c.h"],
  )
  assert (
      set([
          BINDIR.joinpath("ggg/includes"),
          SRCDIR.joinpath("ggg/includes"),
      ])
      == target_includes.system
  )

  # reposnitory-relative
  # bazel doesn't generate this one; it probably builds file symlinks.
  target_includes = construct_cc_includes(
      PackageId("foo", "hhh"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      includes=["/includes"],
      known_include_files=["foo-srcdir/includes/a.h"],
  )
  assert (
      set([
          BINDIR.joinpath("includes"),
          SRCDIR.joinpath("includes"),
      ])
      == target_includes.system
  )


def test_construct_cc_includes_include_prefix():
  # include_prefix really doesn't work for bazel_to_cmake.
  target_includes = construct_cc_includes(
      PackageId("foo", "jjj"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      include_prefix="jjj",
      known_include_files=["foo-srcdir/jjj/x.h"],
  )
  assert (
      set([
          SRCDIR,
      ])
      == target_includes.public
  )

  target_includes = construct_cc_includes(
      PackageId("foo", "kkk"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      include_prefix="_mismatch_",
      known_include_files=["foo-srcdir/kkk/y.h"],
  )
  assert (
      set([
          SRCDIR,
      ])
      == target_includes.public
  )


def test_construct_cc_includes_strip_include_prefix():
  # mismatched
  target_includes = construct_cc_includes(
      PackageId("foo", "lll"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      strip_include_prefix="xyz",
      known_include_files=["foo-srcdir/lll/a.h"],
  )
  assert (
      set([
          SRCDIR,
      ])
      == target_includes.public
  )
  target_includes = construct_cc_includes(
      PackageId("foo", "mmm"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      strip_include_prefix="xyz",
      known_include_files=["foo-bindir/mmm/b.h"],
  )
  assert (
      set([
          BINDIR,
      ])
      == target_includes.public
  )
  # Respoitory relative
  target_includes = construct_cc_includes(
      PackageId("foo", "nnn"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      strip_include_prefix="/nnn",
      known_include_files=["foo-srcdir/nnn/c.h"],
  )
  assert (
      set([
          SRCDIR.joinpath("nnn"),
      ])
      == target_includes.public
  )
  target_includes = construct_cc_includes(
      PackageId("foo", "ooo"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      strip_include_prefix="/ooo",
      known_include_files=["foo-srcdir/ooo/d.h", "foo-bindir/ooo/dd.h"],
  )
  assert (
      set([
          BINDIR.joinpath("ooo"),
          SRCDIR.joinpath("ooo"),
      ])
      == target_includes.public
  )
  target_includes = construct_cc_includes(
      PackageId("foo", "ppp"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      strip_include_prefix="/ppp",
      known_include_files=["foo-srcdir/ppp/e.inc", "foo-srcdir/ppp/e.h"],
  )
  assert (
      set([
          SRCDIR.joinpath("ppp"),
      ])
      == target_includes.public
  )
  # Package includes
  target_includes = construct_cc_includes(
      PackageId("foo", "rrr"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      strip_include_prefix="rrr",
      known_include_files=["foo-srcdir/rrr/f.h"],
  )
  assert (
      set([
          SRCDIR,
      ])
      == target_includes.public
  )
  target_includes = construct_cc_includes(
      PackageId("foo", "sss"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      strip_include_prefix="includes",
      known_include_files=["foo-srcdir/sss/includes/g.h"],
  )
  assert (
      set([
          SRCDIR.joinpath("sss/includes"),
      ])
      == target_includes.public
  )
