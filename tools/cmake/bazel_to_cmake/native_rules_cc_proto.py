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
"""Implementation of cc_proto_library and associated aspects.

The Bazel implementation of cc_proto_library() works closely with the
proto_library() implementation to build C++ sources from protocol buffers.

These implementations use the INTERFACE libraries declared earlier to
generate and compile C++ files into specific targets. These targets
have common names which are derived from the proto_library() targets,
and the cc_proto_library() targets aggregate these together.

Common naming allows cross-project dependencies to "blind" link to the
generated targets if necessary.

This generator assumes that appropriate mappings between bazel and CMake targets
have been configured:
  @com_google_protobuf//:protobuf      => protobuf::libprotobuf
  @com_google_protobuf//:protobuf_lite => protobuf::libprotobuf-lite
  @com_google_protobuf//:protoc        => protobuf::protoc
  @com_google_protobuf//:protoc_lib    => protobuf::libprotoc

In debian-like systems, use of system versions requires the following packages:
  apt install libprotobuf-dev libprotoc-dev protobuf-compiler

Note that when using system protobuf, the well_known_proto_types are available
via protobuf::libprotobuf. For reference, see

Bazel rules related to protobuf for reference:
https://github.com/bazelbuild/bazel/tree/master/src/main/starlark/builtins_bzl/common/proto/proto_library.bzl
https://github.com/bazelbuild/bazel/tree/master/src/main/starlark/builtins_bzl/common/cc/cc_proto_library.bzl
https://github.com/bazelbuild/rules_proto/tree/master/proto

NOTE: The current generator is not well tested with system libraries due to
changes in how upb and protobuf are bundled.
"""

# pylint: disable=invalid-name

from typing import Callable, List, Optional

from .cmake_builder import CMakeBuilder
from .cmake_target import CMakeTarget
from .emit_cc import emit_cc_library
from .evaluation import EvaluationState
from .native_aspect_proto import add_proto_aspect
from .native_aspect_proto import aspect_genproto_library_target
from .native_aspect_proto import PluginSettings
from .starlark.bazel_globals import register_native_build_rule
from .starlark.bazel_target import RepositoryId
from .starlark.bazel_target import TargetId
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.provider import TargetInfo

PROTO_REPO = RepositoryId("com_google_protobuf")
PROTO_RUNTIME = PROTO_REPO.parse_target("//:protobuf")

_WELL_KNOWN_TYPES = [
    "any",
    "api",
    "duration",
    "empty",
    "field_mask",
    "source_context",
    "struct",
    "timestamp",
    "type",
    "wrappers",
]

_IGNORED = set([
    PROTO_REPO.parse_target(f"//src/google/protobuf:{x}_proto")
    for x in _WELL_KNOWN_TYPES
])


_CC = PluginSettings(
    name="cpp",
    plugin=None,
    exts=[".pb.h", ".pb.cc"],
    runtime=[PROTO_RUNTIME],
)


def _cpp_proto_target(t: TargetId) -> TargetId:
  return t.get_target_id(f"{t.target_name}__cpp_library")


def cpp_proto_aspect(
    context: InvocationContext,
    proto_target: TargetId,
    visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  if proto_target in _IGNORED:
    return
  aspect_target = _cpp_proto_target(proto_target)
  context.add_rule(
      aspect_target,
      lambda: aspect_genproto_library_target(
          context,
          target=aspect_target,
          proto_target=proto_target,
          plugin=_CC,
          aspect_dependency=_cpp_proto_target,
          **kwargs,
      ),
      visibility=visibility,
  )


add_proto_aspect("cpp", cpp_proto_aspect)


@register_native_build_rule
def cc_proto_library(
    self: InvocationContext,
    name: str,
    visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  context = self.snapshot()
  target = context.parse_rule_target(name)
  context.add_rule(
      target,
      lambda: cc_proto_library_impl(
          context,
          target,
          _aspect_target=_cpp_proto_target,
          _mnemonic="cc_proto_library",
          **kwargs,
      ),
      visibility=visibility,
  )


def cc_proto_library_impl(
    _context: InvocationContext,
    _target: TargetId,
    *,
    _aspect_target: Callable[[TargetId], TargetId],
    _mnemonic: str,
    deps: Optional[List[RelativeLabel]] = None,  # points to proto_library rules
    **kwargs,
):
  del kwargs

  resolved_deps = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(deps)
  )

  state = _context.access(EvaluationState)
  cmake_target_pair = state.generate_cmake_target_pair(_target)

  # Typically there is a single proto dep in a cc_library_target, multiple are
  # supported, thus we resolve each library target here.
  library_deps: List[CMakeTarget] = state.get_deps(
      [_aspect_target(t) for t in resolved_deps]
  )

  builder = _context.access(CMakeBuilder)
  builder.addtext(f"\n# {_mnemonic}({_target.as_label()})")
  emit_cc_library(
      builder,
      cmake_target_pair,
      hdrs=set(),
      srcs=set(),
      includes=[],
      deps=set(library_deps),
  )
  _context.add_analyzed_target(
      _target, TargetInfo(*cmake_target_pair.as_providers())
  )
