# Copyright 2025 The TensorStore Authors
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

from .parse_bazelrc import ParsedBazelrc
from .starlark.bazel_target import TargetId


def test_parse_bazelrc() -> None:
  parsed = ParsedBazelrc("windows")
  parsed.load_bazelrc_text("""
build --define=foo=bar
build:windows --define=baz=qux
""")
  assert parsed.values == {
      ("define", "foo=bar"),
      ("define", "baz=qux"),
  }


def test_parse_bazelrc_cdefines() -> None:
  parsed = ParsedBazelrc("windows")
  parsed.load_bazelrc_text("""
build:windows --copt=/Dbaz=qux
build:windows --per_file_copt=.*\\\\.cc$@-DFOO=bar
""")
  assert parsed.cdefines == ["baz=qux", "FOO=bar"]


def test_parse_per_file_copt() -> None:
  parsed = ParsedBazelrc("gcc_or_clang")
  parsed.load_bazelrc_text("""
build:gcc_or_clang --per_file_copt=.*\\\\.h$,.*\\\\.cc$,.*\\\\.cpp$@-std=c++17,-fsized-deallocation
build:gcc_or_clang --host_per_file_copt=.*\\\\.h$,.*\\\\.cc$,.*\\\\.cpp$@-std=c++17,-fsized-deallocation

build:gcc_or_clang --per_file_copt=upb/.*\\\\.c$@-Wno-array-bounds,-Wno-stringop-overread
build:gcc_or_clang --per_file_copt=upbc/.*\\\\.cc$@-Wno-array-bounds,-Wno-stringop-overread
build:gcc_or_clang --per_file_copt=grpc/src/.*\\\\.cc$@-Wno-attributes
build:gcc_or_clang --per_file_copt=o@/hasO
""")
  assert parsed.per_file_copt == [
      ("upb/.*\\.c$", "-Wno-array-bounds"),
      ("upb/.*\\.c$", "-Wno-stringop-overread"),
      ("upbc/.*\\.cc$", "-Wno-array-bounds"),
      ("upbc/.*\\.cc$", "-Wno-stringop-overread"),
      ("grpc/src/.*\\.cc$", "-Wno-attributes"),
      ("o", "/hasO"),
  ]

  assert parsed.get_per_file_copts(
      TargetId.parse("@protobuf//upb:foo.c"),
      "external/protobuf/upb/foo.c",
  ) == ["-Wno-array-bounds", "-Wno-stringop-overread", "/hasO"]

  assert parsed.get_per_file_copts(
      TargetId.parse("@protobuf//upbc:foo.cc"),
      "external/protobuf/upbc/foo.cc",
  ) == ["-Wno-array-bounds", "-Wno-stringop-overread", "/hasO"]

  assert parsed.get_per_file_copts(
      TargetId.parse("@grpc//src:foo.c"),
      "external/grpc/src/foo.cc",
  ) == ["-Wno-attributes", "/hasO"]

  assert parsed.get_per_file_copts(
      TargetId.parse("@grpc//:foo.c"),
      "external/grpc/foo.cc",
  ) == ["/hasO"]


def test_parse_copt() -> None:
  parsed = ParsedBazelrc("msvc")
  parsed.load_bazelrc_text("""
build:msvc --copt=-Wa,-mbig-obj
build:msvc --cxxopt=/std:c++17
build:msvc --cxxopt=/Zc:sizedDealloc
build:msvc --conlyopt=/std:c17
""")
  assert parsed.conlyopts == ["/std:c17"]
  assert parsed.copts == ["-Wa,-mbig-obj"]
  assert parsed.cxxopts == ["/Zc:sizedDealloc"]
