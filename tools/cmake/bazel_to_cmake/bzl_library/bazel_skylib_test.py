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

from typing import Any, Dict, List, Optional, Type, TypeVar
import unittest

from ..starlark.bazel_globals import BuildFileGlobals
from ..starlark.bazel_globals import get_bazel_library
from ..starlark.bazel_target import RepositoryId
from ..starlark.bazel_target import TargetId
from ..starlark.ignored import IgnoredLibrary
from ..starlark.invocation_context import InvocationContext
from ..starlark.invocation_context import RuleImpl
from ..starlark.provider import TargetInfo

T = TypeVar("T")

BUILD_BAZEL = """

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag", "string_flag")

"""


class MyRuleContext(InvocationContext):

  def __init__(self):
    self.build_target = TargetId("@bar//:BUILD.bazel")
    self.build_scope = BuildFileGlobals(self, self.build_target, "")
    self.rules: Dict[RuleImpl] = {}
    self.ignored = IgnoredLibrary()

  def access(self, provider_type: Type[T]) -> T:
    return self.ignored

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
    library_type = get_bazel_library((target, False))
    if library_type is not None:
      return library_type(self, target, "builtin")
    return IgnoredLibrary()

  def add_rule(self,
               rule_id: TargetId,
               impl: RuleImpl,
               outs: Optional[List[TargetId]] = None,
               **kwargs) -> None:
    self.rules[rule_id] = impl

  def add_analyzed_target(self, target_id: TargetId, info: TargetInfo) -> None:
    pass


class BazelSkylibTest(unittest.TestCase):

  def test_rule(self):
    self.maxDiff = None  # pylint: disable=invalid-name
    ctx = MyRuleContext()

    # Compile the BUILD file.
    exec(compile(BUILD_BAZEL, "build", "exec"), ctx.build_scope)  # pylint: disable=exec-used
