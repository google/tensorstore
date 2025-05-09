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
"""CMake implementation of "@bazel_tools//tools/build_defs/repo:utils.bzl"."""

# pylint: disable=invalid-name,missing-function-docstring,relative-beyond-top-level,g-long-lambda

from ..starlark.scope_common import ScopeCommon
from .register import register_bzl_library


@register_bzl_library(
    "@bazel_tools//tools/build_defs/repo:utils.bzl", workspace=True
)
class BazelToolsRepoUtilsLibrary(ScopeCommon):

  def bazel_maybe(self, fn, **kwargs):
    fn(**kwargs)
