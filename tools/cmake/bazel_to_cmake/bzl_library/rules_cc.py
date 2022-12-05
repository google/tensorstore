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
"""CMake implementation of "@rules_cc"."""

# pylint: disable=relative-beyond-top-level,missing-class-docstring

from .. import native_rules_cc
from .. import native_rules_proto
from ..starlark.bazel_globals import BazelGlobals
from ..starlark.bazel_globals import register_bzl_library


@register_bzl_library("@rules_cc//cc:defs.bzl", build=True)
class RulesCcDefsLibrary(BazelGlobals):

  def bazel_cc_library(self, **kwargs):
    return native_rules_cc.cc_library(self._context, **kwargs)

  def bazel_cc_binary(self, **kwargs):
    return native_rules_cc.cc_binary(self._context, **kwargs)

  def bazel_cc_test(self, **kwargs):
    return native_rules_cc.cc_test(self._context, **kwargs)

  def bazel_cc_proto_library(self, **kwargs):
    return native_rules_proto.cc_proto_library(self._context, **kwargs)
