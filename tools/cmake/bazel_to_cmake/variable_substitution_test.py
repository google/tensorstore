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

import os
from typing import Optional
import unittest

from .cmake_target import CMakeTargetProvider
from .starlark.bazel_target import PackageId
from .starlark.bazel_target import RepositoryId
from .starlark.bazel_target import TargetId
from .starlark.common_providers import FilesProvider
from .starlark.invocation_context import InvocationContext
from .starlark.provider import TargetInfo
from .starlark.toolchain import CMAKE_TOOLCHAIN
from .variable_substitution import apply_location_substitutions
from .variable_substitution import apply_make_variable_substitutions


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

  def get_target_info(self, target_id: TargetId) -> TargetInfo:
    providers = []
    if "self" in repr(target_id):
      providers.append(FilesProvider([__file__]))
    if "file" in repr(target_id):
      providers.append(
          FilesProvider([f"{target_id.package_name}/{target_id.target_name}"]))
    if "cmake" in repr(target_id):
      providers.append(CMakeTargetProvider(target_id.target_name))
    if "none" in repr(target_id):
      return None
    return TargetInfo(*providers)


class TestSubs(unittest.TestCase):

  def test_apply_make_variable_substitutions(self):
    ctx = MyContext()

    with self.assertRaises(Exception) as _:
      apply_make_variable_substitutions(ctx, "$$(location $(foo)) $@", {}, [])

    subs = {"foo": "bar", "@": "x", "<": "y"}

    self.assertEqual(
        "bar x", apply_make_variable_substitutions(ctx, "$(foo) $@", subs, []))

    self.assertEqual(
        "$(location bar) x",
        apply_make_variable_substitutions(ctx, "$$(location $(foo)) $@", subs,
                                          []))

    self.assertEqual(
        "${bar} x",
        apply_make_variable_substitutions(ctx, "$${bar} $@", subs, []))

    self.assertEqual(
        "${CMAKE_COMMAND} x",
        apply_make_variable_substitutions(
            ctx,
            "$(CMAKE_COMMAND) $@",
            subs,
            toolchains=[TargetId(CMAKE_TOOLCHAIN)]))

  def test_apply_location_substitutions(self):
    ctx = MyContext()
    relative_to = os.path.dirname(__file__)

    # TargetInfo with no targets raises an exception
    with self.assertRaises(Exception) as _:
      apply_location_substitutions(ctx, "$(location a/none/target)", "", [])

    with self.assertRaises(Exception) as _:
      apply_location_substitutions(ctx, "$(location an/empty/target)", "", [])

    self.assertEqual(
        "$(frob my/file/a.h) $@",
        apply_location_substitutions(ctx, "$(frob my/file/a.h) $@", ""))

    self.assertEqual(
        "variable_substitution_test.py $@",
        apply_location_substitutions(ctx, "$(location my/self/a.h) $@",
                                     relative_to))

    self.assertEqual(
        "$<TARGET_FILE:cmake/target> $@",
        apply_location_substitutions(ctx, "$(location cmake/target) $@",
                                     relative_to))
