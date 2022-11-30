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

# pylint: disable=g-importing-member

from typing import Any, Dict, List, Optional
import unittest

from .bazel_globals import BazelWorkspaceGlobals
from .bazel_globals import BuildFileLibraryGlobals
from .bazel_target import PackageId
from .bazel_target import RepositoryId
from .bazel_target import TargetId
from .ignored import IgnoredLibrary
from .invocation_context import InvocationContext
from .invocation_context import RuleImpl

WORKSPACE_GLOBALS = """

load(":b.bzl", "bzzz")

def print_stuff(x):
    print(len([1,2,3]))
    print(all([True, False]))
    print(any([True, False]))
    print(list(sorted([2,1, 3])))
    print(list(reversed([1,2,3])))
    print(depset([1, 2, 3]))
    print(dict([('one', x), (3, 4)]))
    print(list(range(0, 5, 10)))
    print(list(range(5)))
    print(min(5, -2, 1, 7, 3))
    print(max(5, -2, 1, 7, 3))
    print(list(enumerate([False, True, None], 42)))
    print(list(zip([1, 2, 3])))
    print(hasattr("", "find"))
    print(repr(struct(x=1, y=x)))
    a = Label("//:x")
    print(a)
    print(a.name)
    print(a.workspace_name)
    print(a.workspace_root)
"""

WORKSPACE_OUTPUT = """3
False
True
[1, 2, 3]
[3, 2, 1]
depset([1, 2, 3])
{'one': 'x', 3: 4}
[0]
[0, 1, 2, 3, 4]
-2
7
[(42, False), (43, True), (44, None)]
[(1,), (2,), (3,)]
True
"struct(x=1,y='x')"
Label("@foo//:x")
'x'
'foo'
'external/foo'"""


class MyContext(InvocationContext):

  def __init__(self):
    self._caller_package_id = PackageId("@foo//bar/baz")

  @property
  def caller_package_id(self):
    return self._caller_package_id

  def resolve_repo_mapping(
      self, target: TargetId,
      mapping_repository_id: Optional[RepositoryId]) -> TargetId:
    return target

  def resolve_source_root(self, repository_id: RepositoryId) -> str:
    return f"external/{repository_id.repository_name}"

  def load_library(self, target: TargetId) -> Dict[str, Any]:
    return IgnoredLibrary()

  def add_rule(self,
               rule_id: TargetId,
               impl: RuleImpl,
               outs: Optional[List[TargetId]] = None,
               **kwargs) -> None:
    pass


class GlobalsTest(unittest.TestCase):

  def test_workspace_globals(self):
    self.maxDiff = None  # pylint: disable=invalid-name
    scope = BazelWorkspaceGlobals(MyContext(), TargetId("@foo//:WORKSPACE"),
                                  "foo/WORKSPACE")

    output = []
    scope["print"] = output.append

    exec(compile(WORKSPACE_GLOBALS, "foo", "exec"), scope)  # pylint: disable=exec-used

    scope["print_stuff"]("x")
    self.assertEqual("\n".join(repr(x) for x in output), WORKSPACE_OUTPUT)

  def test_build_file_library_globals(self):
    self.maxDiff = None  # pylint: disable=invalid-name
    scope = BuildFileLibraryGlobals(MyContext(), TargetId("@foo//:file.bzl"),
                                    "foo/file.bzl")

    output = []
    scope["print"] = output.append

    exec(compile(WORKSPACE_GLOBALS, "foo", "exec"), scope)  # pylint: disable=exec-used

    scope["print_stuff"]("x")
    self.assertEqual("\n".join(repr(x) for x in output), WORKSPACE_OUTPUT)
