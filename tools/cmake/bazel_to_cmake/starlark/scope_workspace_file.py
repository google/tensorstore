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
"""Starlark globals for CMake."""

# pylint: disable=invalid-name,missing-function-docstring,relative-beyond-top-level,g-importing-member

from .invocation_context import InvocationContext
from .scope_common import ScopeCommon


class BazelNativeWorkspaceRules:
  """Defines the `native` global accessible when evaluating WORKSPACE files."""

  def __init__(self, context: InvocationContext):
    self._context = context

  def bind(self, *args, **kwargs):
    pass

  def existing_rule(self, *args, **kwargs):
    return False


class ScopeWorkspaceFile(ScopeCommon):
  """Globals for WORKSPACE file and .bzl libraries loaded from the WORKSPACE."""

  def bazel_workspace(self, *args, **kwargs):
    pass

  def bazel_register_toolchains(self, *args, **kwargs):
    pass

  @property
  def bazel_native(self):
    return BazelNativeWorkspaceRules(self._context)
