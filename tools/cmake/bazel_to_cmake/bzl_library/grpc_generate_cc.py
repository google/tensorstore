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
"""CMake implementation of "@tensorstore//bazel:cc_grpc_library.bzl"."""

# pylint: disable=invalid-name,missing-function-docstring,relative-beyond-top-level,g-long-lambda
from typing import Any, List, Optional, cast

from ..cmake_builder import CMakeBuilder
from ..cmake_target import CMakeDepsProvider
from ..cmake_target import CMakeTarget
from ..evaluation import EvaluationState
from ..native_aspect_proto import btc_protobuf
from ..native_aspect_proto import get_aspect_dep
from ..native_aspect_proto import PluginSettings
from ..starlark.bazel_globals import BazelGlobals
from ..starlark.bazel_globals import register_bzl_library
from ..starlark.bazel_target import RepositoryId
from ..starlark.bazel_target import TargetId
from ..starlark.common_providers import FilesProvider
from ..starlark.common_providers import ProtoLibraryProvider
from ..starlark.invocation_context import InvocationContext
from ..starlark.invocation_context import RelativeLabel
from ..starlark.provider import TargetInfo
from ..starlark.select import Configurable


GRPC_REPO = RepositoryId("com_github_grpc_grpc")

_SEP = "\n        "

_GRPC = PluginSettings(
    name="grpc",
    plugin=GRPC_REPO.parse_target("//src/compiler:grpc_cpp_plugin"),
    exts=[".grpc.pb.h", ".grpc.pb.cc"],
    runtime=[GRPC_REPO.parse_target("//:grpc++_codegen_proto")],
)


@register_bzl_library(
    "@com_github_grpc_grpc//bazel:generate_cc.bzl", build=True
)
class GrpcGenerateCcLibrary(BazelGlobals):

  def bazel_generate_cc(
      self,
      well_known_protos: Any,
      name: str,
      visibility: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    context = self._context.snapshot()
    target = context.resolve_target(name)
    context.add_rule(
        target,
        lambda: _generate_cc_impl(
            context,
            target,
            well_known_protos=well_known_protos,
            **kwargs,
        ),
        analyze_by_default=False,
    )


def _generate_cc_impl(
    _context: InvocationContext,
    _target: TargetId,
    srcs: Optional[Configurable[List[RelativeLabel]]] = None,
    plugin: Optional[Configurable[RelativeLabel]] = None,
    flags: Optional[List[str]] = None,
    **kwargs,
):
  del kwargs
  state = _context.access(EvaluationState)

  plugin_settings = _GRPC
  if plugin is not None:
    resolved_plugin = _context.resolve_target_or_label(
        cast(RelativeLabel, _context.evaluate_configurable(plugin))
    )
    plugin_settings = PluginSettings(
        name=_GRPC.name,
        plugin=resolved_plugin,
        exts=_GRPC.exts,
        runtime=_GRPC.runtime,
    )

  assert plugin_settings.plugin is not None
  cmake_target_pair = state.generate_cmake_target_pair(_target, alias=False)

  # Only a single source is allowed.
  resolved_srcs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(srcs)
  )
  assert len(resolved_srcs) == 1
  proto_library_target = resolved_srcs[0]

  # Construct the generated paths, installing this rule as a dependency.
  # TODO: Handle skip_import_prefix?
  proto_src_files = []
  generated_paths = []
  cmake_deps: List[CMakeTarget] = [
      get_aspect_dep(
          _context,
          proto_library_target.get_target_id(
              f"{proto_library_target.target_name}__upb_library"
          ),
      )
  ]

  proto_info = _context.get_target_info(proto_library_target).get(
      ProtoLibraryProvider
  )
  assert proto_info is not None

  for src in proto_info.srcs:
    proto_src_files.extend(state.get_file_paths(src, cmake_deps))
    assert src.target_name.endswith(".proto"), f"{repr(src)} must end in .proto"
    proto_prefix = src.target_name[: -len(".proto")]
    for ext in plugin_settings.exts:
      generated_target = src.get_target_id(f"{proto_prefix}{ext}")
      generated_path = _context.get_generated_file_path(generated_target)
      _context.add_analyzed_target(
          generated_target,
          TargetInfo(
              FilesProvider([generated_path]),
              CMakeDepsProvider([cmake_target_pair.target]),
          ),
      )
      generated_paths.append(generated_path)

  _context.add_analyzed_target(
      _target,
      TargetInfo(
          *cmake_target_pair.as_providers(), FilesProvider(generated_paths)
      ),
  )

  out = btc_protobuf(
      _context,
      cmake_target_pair.target,
      proto_library_target,
      plugin_settings,
      cmake_deps=cmake_deps,
      flags=flags,
  )

  builder = _context.access(CMakeBuilder)
  builder.addtext(f"""
# {_target.as_label()}
add_custom_target({cmake_target_pair.target})
{out}
""")
