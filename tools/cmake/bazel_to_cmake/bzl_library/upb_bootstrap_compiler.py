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

"""CMake implementation of "@com_google_protobuf_upb//upbc:bootstrap_compiler.bzl".

bootstrap_compiler.bzl is used to break the otherwise circular dependencies
from the upb build process.
"""

from ..native_rules_cc import cc_binary
from ..native_rules_cc import cc_library
from ..starlark.bazel_globals import BazelGlobals
from ..starlark.bazel_globals import register_bzl_library
from ..starlark.bazel_target import RepositoryId


UPB_REPO = RepositoryId("com_google_protobuf_upb")

_STAGES = ["_stage0", "_stage1", ""]


@register_bzl_library(
    "@com_google_protobuf_upb//upbc:bootstrap_compiler.bzl", build=True
)
class UpbBootstrapCompiler(BazelGlobals):

  def bazel_bootstrap_cc_library(
      self,
      *,
      name,
      visibility=None,
      deps=None,
      bootstrap_deps=None,
      **kwargs,
  ):
    if not deps:
      deps = []
    if not bootstrap_deps:
      bootstrap_deps = []

    for stage in _STAGES:
      cc_library(
          self._context,
          name=name + stage,
          deps=deps + [dep + stage for dep in bootstrap_deps],
          visibility=visibility if not stage else ["//upbc:__pkg__"],
          **kwargs,
      )

  def bazel_bootstrap_cc_binary(
      self,
      *,
      name,
      deps=None,
      bootstrap_deps=None,
      **kwargs,
  ):
    if not deps:
      deps = []
    if not bootstrap_deps:
      bootstrap_deps = []
    for stage in _STAGES:
      cc_binary(
          self._context,
          name=name + stage,
          deps=deps + [dep + stage for dep in bootstrap_deps],
          **kwargs,
      )

  def bazel_bootstrap_upb_proto_library(
      self,
      *,
      name,
      **kwargs,
  ):
    assert name == "descriptor_upb_proto" or name == "plugin_upb_proto"
    # These are actually handled in cmake_extra.BUILD.bazel file.
    return
