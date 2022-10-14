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
"""CMake implementation of "@rules_proto"."""

# pylint: disable=relative-beyond-top-level

from .. import native_rules
from ..evaluation import BazelGlobals
from ..evaluation import IgnoredObject
from ..evaluation import register_bzl_library


@register_bzl_library("@rules_proto//proto:defs.bzl", build=True)
class RulesCcDefsLibrary(BazelGlobals):

  def bazel_proto_library(self, **kwargs):
    return native_rules.proto_library(self._context, **kwargs)

  @property
  def bazel_proto_lang_toolchain(self):
    return IgnoredObject()
