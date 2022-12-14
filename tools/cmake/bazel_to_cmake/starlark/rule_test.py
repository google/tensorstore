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

# pylint: disable=g-importing-member,wildcard-import

from typing import Any, Dict, List, Optional
import unittest

from .bazel_globals import BuildFileGlobals
from .bazel_globals import BuildFileLibraryGlobals
from .bazel_target import RepositoryId
from .bazel_target import TargetId
from .ignored import IgnoredLibrary
from .invocation_context import InvocationContext
from .invocation_context import RuleImpl
from .provider import TargetInfo
from .rule import *

RULE_BZL = """

def _empty_rule_impl(ctx):
  print("empty_rule", ctx.label)

empty_rule = rule(
    implementation = _empty_rule_impl,
    attrs = {
        "a_string": attr.string(default = "b"),
        "a_label": attr.label(allow_files = [".header"]),
        "a_labellist": attr.label_list(),
        "a_bool": attr.bool(default = False),
        "out": attr.output(),
    },
)

"""

BUILD_BAZEL = """

load("@foo//:rule.bzl", "empty_rule")

empty_rule(name = "nothing",
           a_string = "c",
           a_label = Label("//:nothing"),
           a_labellist = [])

"""

OUTPUT = """
'empty_rule'
Label("@bar//:nothing")
"""


class MyRuleContext(InvocationContext):

  def __init__(self):
    self.output = []
    self.rule_target = TargetId("@foo//:rule.bzl")
    self.rule_scope = BuildFileLibraryGlobals(self, self.rule_target, "")
    self.rule_scope["print"] = self._my_print
    self.build_target = TargetId("@bar//:BUILD.bazel")
    self.build_scope = BuildFileGlobals(self, self.build_target, "")
    self.build_scope["print"] = self._my_print
    self.rules: Dict[RuleImpl] = {}

  def _my_print(self, *args):
    for x in args:
      self.output.append(x)

  @property
  def joined_output(self):
    return "\n" + "\n".join(repr(x) for x in self.output) + "\n"

  def snapshot(self) -> "MyRuleContext":
    return self

  @property
  def caller_package_id(self):
    return self.build_target.package_id

  def resolve_repo_mapping(
      self, target: TargetId,
      mapping_repository_id: Optional[RepositoryId]) -> TargetId:
    return target

  def resolve_workspace_root(self, repository_id: RepositoryId) -> str:
    return f"external/{repository_id.repository_name}"

  def load_library(self, target: TargetId) -> Dict[str, Any]:
    if target == self.rule_target:
      return self.rule_scope
    else:
      return IgnoredLibrary()

  def add_rule(self,
               rule_id: TargetId,
               impl: RuleImpl,
               outs: Optional[List[TargetId]] = None,
               **kwargs) -> None:
    self.rules[rule_id] = impl

  def add_analyzed_target(self, target_id: TargetId, info: TargetInfo) -> None:
    pass

  def get_target_info(self, target_id: TargetId) -> TargetInfo:
    return TargetInfo()


class RuleTest(unittest.TestCase):

  def test_rule(self):
    self.maxDiff = None  # pylint: disable=invalid-name
    ctx = MyRuleContext()

    # Compile the .bzl library
    exec(compile(RULE_BZL, "rule", "exec"), ctx.rule_scope)  # pylint: disable=exec-used

    # Compile the BUILD file.
    exec(compile(BUILD_BAZEL, "build", "exec"), ctx.build_scope)  # pylint: disable=exec-used

    for _, impl in ctx.rules.items():
      impl()

    self.assertEqual(ctx.joined_output, OUTPUT)
