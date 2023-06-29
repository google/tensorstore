# Copyright 2023 The TensorStore Authors
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

# pylint: disable=g-importing-member
import pathlib

from .cmake_repository import CMakeRepository
from .cmake_repository import make_repo_mapping
from .cmake_target import CMakePackage
from .starlark.bazel_target import RepositoryId


def test_ignored():
  x = CMakeRepository(
      RepositoryId("foo"),
      CMakePackage("Bar"),
      pathlib.PurePath("src"),
      pathlib.PurePath("bin"),
      {},
      {},
  )
  assert x.repository_id == RepositoryId("foo")

  assert x.get_source_file_path(
      x.repository_id.parse_target("//file")
  ) == pathlib.PurePath("src/file/file")
  assert x.get_generated_file_path(
      x.repository_id.parse_target("//:file")
  ) == pathlib.PurePath("bin/file")

  assert list(make_repo_mapping(x.repository_id, {"@a": "@b"}).items()) == [(
      RepositoryId("a"),
      RepositoryId("b"),
  )]

  assert list(make_repo_mapping(x.repository_id, [("@c", "@d")]).items()) == [(
      RepositoryId("c"),
      RepositoryId("d"),
  )]
