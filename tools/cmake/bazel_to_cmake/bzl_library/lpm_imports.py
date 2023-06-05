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
"""CMake implementation of "@local_proto_mirror//:imports.bzl"."""

# NOTE: It would be nice to make this obsolete;  to do that bazel_to_cmake
# would need to change how it resolves or references repositories.

from .. import native_rules_proto
from ..starlark.bazel_globals import BazelGlobals
from ..starlark.bazel_globals import register_bzl_library
from ..starlark.ignored import IgnoredObject


@register_bzl_library("@local_proto_mirror//:imports.bzl", build=True)
class LocalProtoMirrorImportsLibrary(BazelGlobals):

  def __missing__(self, key):
    if not hasattr(self, f"bazel_{key}"):
      return IgnoredObject()
    return getattr(self, f"bazel_{key}")

  def bazel_cc_proto_library(self, **kwargs):
    return native_rules_proto.cc_proto_library(self._context, **kwargs)

  def bazel_cc_grpc_library(self, name, **kwargs):
    context = self._context.snapshot()
    library_target = context.resolve_target_or_label(
        "@com_google_tensorstore//bazel:cc_grpc_library.bzl"
    )
    library = context.load_library(library_target)
    library.get("cc_grpc_library", None)(name=name, **kwargs)
