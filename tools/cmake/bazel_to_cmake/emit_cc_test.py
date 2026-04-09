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


def test_construct_cc_includes_no_includes():
  target_includes = construct_cc_includes(
      PackageId("foo", "aaa"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
  )
  assert not target_includes.system
  assert not target_includes.public
  assert not target_includes.private


def test_construct_cc_includes_hdrs_include_paths():
  target_includes = construct_cc_includes(
      PackageId("foo", "bbb"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      hdrs_include_paths=["foo-srcdir/bbb/a.inc"],
  )
  assert set([SRCDIR]) == target_includes.public
  assert not target_includes.private

  target_includes = construct_cc_includes(
      PackageId("foo", "ccc"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      hdrs_include_paths=["foo-bindir/ccc/b.inc"],
  )
  assert set([BINDIR]) == target_includes.public
  assert set() == target_includes.private

  # Other combinations
  target_includes = construct_cc_includes(
      PackageId("foo", "ddd"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      hdrs_include_paths=[
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


def test_construct_cc_includes_srcs_include_paths():
  target_includes = construct_cc_includes(
      PackageId("foo", "bbx"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      srcs_file_paths=["foo-srcdir/bbx/a.inc"],
  )
  assert not target_includes.system
  assert not target_includes.public
  assert set([SRCDIR]) == target_includes.private

  target_includes = construct_cc_includes(
      PackageId("foo", "ccx"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      srcs_file_paths=["foo-bindir/ccx/b.inc"],
  )
  assert set() == target_includes.public
  assert set([BINDIR]) == target_includes.private

  # Other combinations
  target_includes = construct_cc_includes(
      PackageId("foo", "ddx"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      srcs_file_paths=[
          "foo-srcdir/ddx/e.inc",
          "foo-bindir/ddx/ee.h",
      ],
  )
  assert (
      set([
          BINDIR,
          SRCDIR,
      ])
      == target_includes.private
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
      hdrs_include_paths=["foo-srcdir/includes/b.h"],
  )
  assert (
      set([
          BINDIR.joinpath("fff/includes"),
          SRCDIR.joinpath("fff/includes"),
      ])
      == target_includes.system
  )
  assert (
      set([
          SRCDIR,
      ])
      == target_includes.public
  )
  assert set() == target_includes.private

  target_includes = construct_cc_includes(
      PackageId("foo", "ggg"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      includes=["includes"],
      hdrs_include_paths=["foo-srcdir/ggg/includes/c.h"],
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
      hdrs_include_paths=["foo-srcdir/includes/a.h"],
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
      hdrs_include_paths=["foo-srcdir/jjj/x.h"],
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
      hdrs_include_paths=["foo-srcdir/kkk/y.h"],
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
      hdrs_include_paths=["foo-srcdir/lll/a.h"],
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
      hdrs_include_paths=["foo-bindir/mmm/b.h"],
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
      hdrs_include_paths=["foo-srcdir/nnn/c.h"],
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
      hdrs_include_paths=["foo-srcdir/ooo/d.h", "foo-bindir/ooo/dd.h"],
  )
  assert (
      set([
          BINDIR.joinpath("ooo"),
          SRCDIR.joinpath("ooo"),
      ])
      == target_includes.public
  )
  assert (
      set([
          SRCDIR,
          BINDIR,
      ])
      == target_includes.private
  )

  target_includes = construct_cc_includes(
      PackageId("foo", "ppp"),
      source_directory=SRCDIR,
      cmake_binary_dir=BINDIR,
      strip_include_prefix="/ppp",
      hdrs_include_paths=["foo-srcdir/ppp/e.inc", "foo-srcdir/ppp/e.h"],
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
      hdrs_include_paths=["foo-srcdir/rrr/f.h"],
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
      hdrs_include_paths=["foo-srcdir/sss/includes/g.h"],
  )
  assert (
      set([
          SRCDIR.joinpath("sss/includes"),
      ])
      == target_includes.public
  )
