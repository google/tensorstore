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
"""CMake implementation of "@com_google_upb//bazel:upb_proto_library.bzl".

https://github.com/protocolbuffers/upb/blob/main/bazel/upb_proto_library.bzl
"""

# pylint: disable=relative-beyond-top-level
from typing import List, Optional

from ..cmake_builder import CMakeBuilder
from ..emit_cc import emit_cc_library
from ..evaluation import EvaluationState
from ..native_rules_proto import get_proto_plugin_library_target
from ..native_rules_proto import PluginSettings
from ..starlark.bazel_globals import BazelGlobals
from ..starlark.bazel_globals import register_bzl_library
from ..starlark.bazel_target import TargetId
from ..starlark.common_providers import BuildSettingProvider
from ..starlark.invocation_context import InvocationContext
from ..starlark.invocation_context import RelativeLabel
from ..starlark.provider import Provider
from ..starlark.provider import TargetInfo
from ..starlark.select import Configurable

_UPB = PluginSettings(
    TargetId("@com_google_upb//upbc:protoc-gen-upb"), "upb",
    [".upb.h", ".upb.c"], [
        TargetId(
            "@com_google_upb//:generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        ),
        TargetId("@com_google_upb//:port")
    ])

_UPBDEFS = PluginSettings(
    TargetId("@com_google_upb//upbc:protoc-gen-upbdefs"), "upbdefs",
    [".upbdefs.h", ".upbdefs.c"], [
        TargetId(
            "@com_google_upb//:generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        ),
        TargetId("@com_google_upb//:port")
    ])


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


@register_bzl_library(
    "@com_google_upb//bazel:upb_proto_library.bzl", build=True)
class UpbProtoLibrary(BazelGlobals):

  def bazel_upb_proto_library(self,
                              name: str,
                              visibility: Optional[List[RelativeLabel]] = None,
                              **kwargs):
    context = self._context.snapshot()
    target = context.resolve_target(name)
    context.add_rule(
        target,
        lambda: _upb_proto_impl(context, target, False, **kwargs),
        visibility=visibility)

  def bazel_upb_proto_reflection_library(
      self,
      name: str,
      visibility: Optional[List[RelativeLabel]] = None,
      **kwargs):
    context = self._context.snapshot()
    target = context.resolve_target(name)
    context.add_rule(
        target,
        lambda: _upb_proto_impl(context, target, True, **kwargs),
        visibility=visibility)

  bazel__FastTableEnabledInfo = staticmethod(_FastTableEnabledInfo)

  def bazel_upb_fasttable_enabled(self, name: str, **kwargs):
    # Really a proxy for bool_flag, but just set it to False.
    del kwargs
    context = self._context.snapshot()
    target = context.resolve_target(name)

    def impl():
      context.add_analyzed_target(
          target,
          TargetInfo(_FastTableEnabledInfo(False), BuildSettingProvider(False)))

    context.add_rule(target, impl, analyze_by_default=True)

  bazel_UpbProtoLibraryCoptsInfo = staticmethod(UpbProtoLibraryCoptsInfo)

  def bazel_upb_proto_library_copts(self, name: str,
                                    copts: Configurable[List[str]], **kwargs):
    # This rule just exists to provide copts to aspects
    del kwargs
    context = self._context.snapshot()
    target = context.resolve_target(name)

    def impl():
      resolved_copts = context.evaluate_configurable_list(copts)
      context.add_analyzed_target(
          target, TargetInfo(UpbProtoLibraryCoptsInfo(resolved_copts)))

    context.add_rule(target, impl, analyze_by_default=True)


def _upb_proto_impl(_context: InvocationContext, _target: TargetId,
                    build_upbdeps: bool, deps: List[RelativeLabel], **kwargs):
  del kwargs
  resolved_deps = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(deps))

  state = _context.access(EvaluationState)
  cmake_target_pair = state.generate_cmake_target_pair(_target)

  # Typically there is a single proto dep in a cc_library_target, multiple are
  # supported, thus we resolve each library target here.
  library_deps = []
  for dep_target in resolved_deps:
    lib_target = get_proto_plugin_library_target(
        _context, plugin_settings=_UPB, target=dep_target)
    library_deps.extend(state.get_dep(lib_target, alias=False))
  if build_upbdeps:
    for dep_target in resolved_deps:
      lib_target = get_proto_plugin_library_target(
          _context, plugin_settings=_UPBDEFS, target=dep_target)
      library_deps.extend(state.get_dep(lib_target, alias=False))

  builder = _context.access(CMakeBuilder)
  builder.addtext(f"\n# {_target.as_label()}")

  emit_cc_library(
      builder,
      cmake_target_pair,
      hdrs=set(),
      srcs=set(),
      deps=set(library_deps),
  )
  _context.add_analyzed_target(_target,
                               TargetInfo(*cmake_target_pair.as_providers()))
