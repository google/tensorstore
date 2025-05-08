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
"""CMake implementation of proto rules from "@com_google_protobuf".

NOTE: The upb rules are not included here.
"""

# pylint: disable=relative-beyond-top-level

from .. import native_rules_cc_proto
from .. import native_rules_proto
from ..starlark.scope_common import ScopeCommon
from .register import ignore_bzl_library
from .register import register_bzl_library

for x in (
    "//bazel:java_lite_proto_library.bzl",
    "//bazel:java_proto_library.bzl",
    "//bazel:py_proto_library.bzl",
):
  ignore_bzl_library(f"@com_google_protobuf{x}")


@register_bzl_library("@com_google_protobuf//bazel:proto_library.bzl")
class ProtobufProtoLibrary(ScopeCommon):

  def bazel_proto_library(self, name: str, **kwargs):
    return native_rules_proto.proto_library(self._context, name, **kwargs)


@register_bzl_library("@com_google_protobuf//bazel:cc_proto_library.bzl")
class ProtobufCcProtoLibrary(ScopeCommon):

  def bazel_cc_proto_library(self, name: str, **kwargs):
    return native_rules_cc_proto.cc_proto_library(self._context, name, **kwargs)
