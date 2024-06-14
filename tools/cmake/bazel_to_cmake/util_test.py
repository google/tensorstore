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
import pathlib

from .util import cmake_is_true
from .util import is_relative_to
from .util import map_path_prefixes
from .util import quote_list
from .util import quote_path_list


def test_quote_list():
  assert quote_list(["a", "b", "c"]) == '"a" "b" "c"'
  assert quote_list(["a", "b", "c"], separator=" ") == '"a" "b" "c"'
  assert quote_list(["a", "b", "c"], separator="\\n") == '"a"\\n"b"\\n"c"'


def test_quote_pathlist():
  assert (
      quote_path_list(
          ["a", pathlib.PureWindowsPath("b\\d"), pathlib.PurePosixPath("c")]
      )
      == '"a" "b/d" "c"'
  )
  assert (
      quote_path_list(
          ["a", pathlib.PureWindowsPath("b\\d"), pathlib.PurePosixPath("c")],
          separator=" ",
      )
      == '"a" "b/d" "c"'
  )


def test_cmake_is_true():
  assert cmake_is_true("1")
  assert cmake_is_true("ON")
  assert cmake_is_true("YES")
  assert cmake_is_true("TRUE")
  assert cmake_is_true("Y")
  assert cmake_is_true("2")
  assert cmake_is_true("notfound")
  assert not cmake_is_true("0")
  assert not cmake_is_true("OFF")
  assert not cmake_is_true("NO")
  assert not cmake_is_true("FALSE")
  assert not cmake_is_true("N")
  assert not cmake_is_true("IGNORE")
  assert not cmake_is_true("NOTFOUND")
  assert not cmake_is_true("x-NOTFOUND")
  assert not cmake_is_true("")
  assert not cmake_is_true(None)


def test_is_relative_to():
  root = pathlib.PurePath("/foo/bar")
  leaf = pathlib.PurePath("/foo/bar/baz")

  assert is_relative_to(leaf, root, _use_attr=False)
  assert not is_relative_to(root, leaf, _use_attr=False)


def test_map_path_prefixes():

  assert map_path_prefixes(
      [
          pathlib.PurePath("/foo/bar"),
          pathlib.PurePath("/foo/bar/baz"),
          "/foo/bindir/baz/blah",
          "/foo/bindir/baz/xyz",
          "/foo/srcdir/baz/blah",
          "/foo/srcdir/baz/xyz",
      ],
      [
          (pathlib.PurePath("/foo/bindir"), "${CMAKE_BINDIR}/"),
          (pathlib.PurePath("/foo/srcdir"), ""),
          (pathlib.PurePath("/bar"), "{BAR}"),
      ],
  ) == [
      pathlib.PurePath("/foo/bar"),
      pathlib.PurePath("/foo/bar/baz"),
      pathlib.PurePath("${CMAKE_BINDIR}/baz/blah"),
      pathlib.PurePath("${CMAKE_BINDIR}/baz/xyz"),
      pathlib.PurePath("baz/blah"),
      pathlib.PurePath("baz/xyz"),
  ]
