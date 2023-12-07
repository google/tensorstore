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
"""CMake implementation of "@com_google_protobuf//bazel:upb_proto_library.bzl".

https://github.com/protocolbuffers/protobuf/blob/main/bazel/upb_proto_library.bzl
https://github.com/protocolbuffers/protobuf/blob/main/bazel/upb_proto_reflection_library.bzl
https://github.com/protocolbuffers/protobuf/blob/main/bazel/upb_proto_library.bzl
"""

# pylint: disable=relative-beyond-top-level
from typing import List, Optional

from ..native_aspect_proto import add_proto_aspect
from ..native_aspect_proto import aspect_genproto_library_target
from ..native_aspect_proto import PluginSettings
from ..native_rules_cc_proto import cc_proto_library_impl
from ..starlark.bazel_globals import BazelGlobals
from ..starlark.bazel_globals import register_bzl_library
from ..starlark.bazel_target import RepositoryId
from ..starlark.bazel_target import TargetId
from ..starlark.invocation_context import InvocationContext
from ..starlark.invocation_context import RelativeLabel
from ..starlark.provider import Provider


UPB_REPO = RepositoryId("com_google_protobuf")

# TODO: Better toolchain support.
# https://github.com/protocolbuffers/protobuf/blob/5ce86a675ea4cfc9dcfc54a9be6141ea6bc371b6/upb_generator/BUILD#L322
#
# proto_lang_toolchain(
#    name = "protoc-gen-upb_minitable_toolchain",
#    command_line = "--upb_minitable_out=$(OUT)",
#    output_files = "multiple",
#    plugin = ":protoc-gen-upb_minitable_stage1",
#    plugin_format_flag = "--plugin=protoc-gen-upb_minitable=%s",
#    progress_message = "Generating upb minitables",
#    runtime = "//upb:generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me",
#    visibility = ["//visibility:public"],
# )
#
# proto_lang_toolchain(
#    name = "protoc-gen-upb_toolchain",
#    command_line = "--upb_out=$(OUT)",
#    output_files = "multiple",
#    plugin = ":protoc-gen-upb_stage1",
#    plugin_format_flag = "--plugin=protoc-gen-upb=%s",
#    progress_message = "Generating upb protos",
#    runtime = "//upb:generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me",
#   visibility = ["//visibility:public"],
# )

_UPB_MINITABLE = PluginSettings(
    name="upb_minitable",
    plugin=UPB_REPO.parse_target(
        "//upb_generator:protoc-gen-upb_minitable_stage1"
    ),
    exts=[".upb_minitable.h", ".upb_minitable.c"],
    runtime=[
        UPB_REPO.parse_target(
            "//upb:generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        ),
    ],
)

_UPB = PluginSettings(
    name="upb",
    plugin=UPB_REPO.parse_target("//upb_generator:protoc-gen-upb"),
    exts=[".upb.h", ".upb.c"],
    runtime=[
        UPB_REPO.parse_target(
            "//upb:generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        ),
    ],
)

# STAGE1 is used for bootstrapping upb via cmake.
_UPB_STAGE1 = PluginSettings(
    name="upb",
    plugin=UPB_REPO.parse_target("//upb_generator:protoc-gen-upb_stage1"),
    exts=[".upb.h", ".upb.c"],
    runtime=[
        UPB_REPO.parse_target(
            "//upb:generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        ),
    ],
)


_UPBDEFS = PluginSettings(
    name="upbdefs",
    plugin=UPB_REPO.parse_target("//upb_generator:protoc-gen-upbdefs"),
    exts=[".upbdefs.h", ".upbdefs.c"],
    runtime=[
        UPB_REPO.parse_target(
            "//upb:generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        ),
        UPB_REPO.parse_target("//upb:port"),
    ],
)


def _minitable_target(t: TargetId) -> TargetId:
  return t.get_target_id(f"{t.target_name}__minitable_library")


def _upb_target(t: TargetId) -> TargetId:
  return t.get_target_id(f"{t.target_name}__upb_library")


def _upbdefs_target(t: TargetId) -> TargetId:
  return t.get_target_id(f"{t.target_name}__upbdefs_library")


def upb_minitable_aspect(
    context: InvocationContext,
    proto_target: TargetId,
    visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  aspect_target = _minitable_target(proto_target)
  context.add_rule(
      aspect_target,
      lambda: aspect_genproto_library_target(
          context,
          target=aspect_target,
          proto_target=proto_target,
          plugin=_UPB_MINITABLE,
          aspect_dependency=_minitable_target,
          **kwargs,
      ),
      visibility=visibility,
  )


def upb_aspect(
    context: InvocationContext,
    proto_target: TargetId,
    visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  plugin = _UPB
  if proto_target.repository_id == UPB_REPO:
    plugin = _UPB_STAGE1

  aspect_target = _upb_target(proto_target)
  minitable_target = _minitable_target(proto_target)
  context.add_rule(
      aspect_target,
      lambda: aspect_genproto_library_target(
          context,
          target=aspect_target,
          proto_target=proto_target,
          plugin=plugin,
          aspect_dependency=_upb_target,
          extra_deps=[minitable_target],
          **kwargs,
      ),
      visibility=visibility,
  )


def upbdefs_aspect(
    context: InvocationContext,
    proto_target: TargetId,
    visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  aspect_target = _upbdefs_target(proto_target)
  minitable_target = _minitable_target(proto_target)
  context.add_rule(
      aspect_target,
      lambda: aspect_genproto_library_target(
          context,
          target=aspect_target,
          proto_target=proto_target,
          plugin=_UPBDEFS,
          aspect_dependency=_upbdefs_target,
          extra_deps=[minitable_target],
          **kwargs,
      ),
      visibility=visibility,
  )


add_proto_aspect("upb", upb_aspect)
add_proto_aspect("upb_minitable", upb_minitable_aspect)
add_proto_aspect("upbdefs", upbdefs_aspect)

#############################################################################


class UpbMinitableCcInfo(Provider):
  __slots__ = tuple("cc_info")

  def __init__(self, cc_info: List[str]):
    self.cc_info = cc_info

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.cc_info)})"


@register_bzl_library(
    "@com_google_protobuf//bazel:upb_minitable_proto_library.bzl", build=True
)
class UpbMinitableProtoLibrary(BazelGlobals):

  # pylint: disable-next=invalid-name
  bazel_UpbMinitableCcInfo = UpbMinitableCcInfo

  def bazel_upb_minitable_proto_library(
      self,
      name: str,
      visibility: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    context = self._context.snapshot()
    target = context.resolve_target(name)
    context.add_rule(
        target,
        lambda: cc_proto_library_impl(
            context,
            target,
            _aspect_target=_minitable_target,
            _mnemonic="upb_minitable_proto_library",
            **kwargs,
        ),
        visibility=visibility,
    )


#############################################################################


class UpbWrappedCcInfo(Provider):
  """Build setting value (i.e. flag value) corresponding to a Bazel target."""

  __slots__ = ("cc_info", "cc_info_with_thunks")

  def __init__(self, cc_info: List[str], cc_info_with_thunks: List[str]):
    self.cc_info = cc_info
    self.cc_info_with_thunks = cc_info_with_thunks

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.cc_info)},{repr(self.cc_info_with_thunks)})"


@register_bzl_library(
    "@com_google_protobuf//bazel:upb_c_proto_library.bzl", build=True
)
class UpbCProtoLibrary(BazelGlobals):

  # pylint: disable-next=invalid-name
  bazel_UpbWrappedCcInfo = UpbWrappedCcInfo

  def bazel_upb_c_proto_library(
      self,
      name: str,
      visibility: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    context = self._context.snapshot()
    target = context.resolve_target(name)
    context.add_rule(
        target,
        lambda: cc_proto_library_impl(
            context,
            target,
            _aspect_target=_minitable_target,
            _mnemonic="upb_c_proto_library",
            **kwargs,
        ),
        visibility=visibility,
    )


#############################################################################


@register_bzl_library(
    "@com_google_protobuf//bazel:upb_proto_reflection_library.bzl", build=True
)
class UpbProtoReflectionLibrary(BazelGlobals):

  def bazel_upb_proto_reflection_library(
      self,
      name: str,
      visibility: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    context = self._context.snapshot()
    target = context.resolve_target(name)
    context.add_rule(
        target,
        lambda: cc_proto_library_impl(
            context,
            target,
            _aspect_target=_upbdefs_target,
            _mnemonic="upb_proto_reflection_library",
            **kwargs,
        ),
        visibility=visibility,
    )


#############################################################################


class UpbProtoLibraryCoptsInfo(Provider):
  """Build setting value (i.e. flag value) corresponding to a Bazel target."""

  __slots__ = ("copts",)

  def __init__(self, copts: List[str]):
    self.copts = copts

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.copts)})"


@register_bzl_library(
    "@com_google_protobuf//bazel:upb_proto_library.bzl", build=True
)
class UpbProtoLibrary(BazelGlobals):

  # pylint: disable-next=invalid-name
  bazel_UpbWrappedCcInfo = UpbWrappedCcInfo

  def bazel_upb_proto_library(
      self,
      name: str,
      visibility: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    context = self._context.snapshot()
    target = context.resolve_target(name)
    context.add_rule(
        target,
        lambda: cc_proto_library_impl(
            context,
            target,
            _aspect_target=_upb_target,
            _mnemonic="upb_proto_library",
            **kwargs,
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
    context.add_rule(
        target,
        lambda: cc_proto_library_impl(
            context,
            target,
            _aspect_target=_upbdefs_target,
            _mnemonic="upb_proto_reflection_library",
            **kwargs,
        ),
        visibility=visibility,
    )
