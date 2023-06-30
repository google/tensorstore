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
"""CMake implementation of "@com_google_protobuf_upb//bazel:upb_proto_library.bzl".

https://github.com/protocolbuffers/upb/blob/main/bazel/upb_proto_library.bzl
"""

# pylint: disable=relative-beyond-top-level
from typing import List, Optional

from ..native_rules_cc_proto import cc_proto_library_impl
from ..native_rules_cc_proto import PluginSettings
from ..starlark.bazel_globals import BazelGlobals
from ..starlark.bazel_globals import register_bzl_library
from ..starlark.bazel_target import RepositoryId
from ..starlark.common_providers import BuildSettingProvider
from ..starlark.invocation_context import RelativeLabel
from ..starlark.provider import Provider
from ..starlark.provider import TargetInfo
from ..starlark.select import Configurable


UPB_REPO = RepositoryId("com_google_protobuf_upb")

# UPB_STAGE1 is used for bootstrapping upb via cmake.
_UPB_STAGE1 = PluginSettings(
    name="upb",
    plugin=UPB_REPO.parse_target("//upbc:protoc-gen-upb_stage1"),
    exts=[".upb.h", ".upb.c"],
    runtime=[
        UPB_REPO.parse_target(
            "//:generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        ),
    ],
    replacement_targets={},
)

_UPB = PluginSettings(
    name="upb",
    plugin=UPB_REPO.parse_target("//upbc:protoc-gen-upb"),
    exts=[".upb.h", ".upb.c"],
    runtime=[
        UPB_REPO.parse_target(
            "//:generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        ),
    ],
    replacement_targets={},
)

_UPBDEFS = PluginSettings(
    name="upbdefs",
    plugin=UPB_REPO.parse_target("//upbc:protoc-gen-upbdefs"),
    exts=[".upbdefs.h", ".upbdefs.c"],
    runtime=[
        UPB_REPO.parse_target(
            "//:generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        ),
        UPB_REPO.parse_target("//:port"),
    ],
    replacement_targets={},
)


class _FastTableEnabledInfo(Provider):
  """Build setting value (i.e. flag value) corresponding to a Bazel target."""

  __slots__ = ("enabled",)

  def __init__(self, enabled: bool):
    self.enabled = enabled

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.enabled)})"


class UpbProtoLibraryCoptsInfo(Provider):
  """Build setting value (i.e. flag value) corresponding to a Bazel target."""

  __slots__ = ("copts",)

  def __init__(self, copts: List[str]):
    self.copts = copts

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.copts)})"


class UpbWrappedCcInfo(Provider):
  """Build setting value (i.e. flag value) corresponding to a Bazel target."""

  __slots__ = ("cc_info", "cc_info_with_thunks")

  def __init__(self, cc_info: List[str], cc_info_with_thunks: List[str]):
    self.cc_info = cc_info
    self.cc_info_with_thunks = cc_info_with_thunks

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.cc_info)},{repr(self.cc_info_with_thunks)})"


@register_bzl_library(
    "@com_google_protobuf_upb//bazel:upb_proto_library.bzl", build=True
)
class UpbProtoLibrary(BazelGlobals):

  # pylint: disable-next=invalid-name
  bazel__FastTableEnabledInfo = _FastTableEnabledInfo

  # pylint: disable-next=invalid-name
  bazel_UpbWrappedCcInfo = UpbWrappedCcInfo

  # pylint: disable-next=invalid-name
  bazel_UpbProtoLibraryCoptsInfo = UpbProtoLibraryCoptsInfo

  def bazel_upb_proto_library(
      self,
      name: str,
      visibility: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    context = self._context.snapshot()
    target = context.resolve_target(name)

    plugin = _UPB
    if target.repository_id == UPB_REPO:
      plugin = _UPB_STAGE1

    context.add_rule(
        target,
        lambda: cc_proto_library_impl(
            context, target, plugin, "upb_proto_library", **kwargs
        ),
        visibility=visibility,
    )

  def bazel_upb_proto_reflection_library(
      self,
      name: str,
      visibility: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    context = self._context.snapshot()
    target = context.resolve_target(name)

    dep_plugin = _UPB
    if target.repository_id == UPB_REPO:
      dep_plugin = _UPB_STAGE1

    context.add_rule(
        target,
        lambda: cc_proto_library_impl(
            context,
            target,
            _UPBDEFS,
            "upb_proto_reflection_library",
            _dep_plugin=dep_plugin,
            **kwargs,
        ),
        visibility=visibility,
    )

  def bazel_upb_fasttable_enabled(self, name: str, **kwargs):
    # Really a proxy for bool_flag, but just set it to False.
    del kwargs
    context = self._context.snapshot()
    target = context.resolve_target(name)

    def impl():
      context.add_analyzed_target(
          target,
          TargetInfo(_FastTableEnabledInfo(False), BuildSettingProvider(False)),
      )

    context.add_rule(target, impl, analyze_by_default=True)

  def bazel_upb_proto_library_copts(
      self, name: str, copts: Configurable[List[str]], **kwargs
  ):
    # This rule just exists to provide copts to aspects
    del kwargs
    context = self._context.snapshot()
    target = context.resolve_target(name)

    def impl():
      resolved_copts = context.evaluate_configurable_list(copts)
      context.add_analyzed_target(
          target, TargetInfo(UpbProtoLibraryCoptsInfo(resolved_copts))
      )

    context.add_rule(target, impl, analyze_by_default=True)
