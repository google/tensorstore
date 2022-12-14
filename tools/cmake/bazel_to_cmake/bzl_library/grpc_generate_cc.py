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
"""CMake implementation of "@com_google_tensorstore//bazel:cc_grpc_library.bzl".
"""

# pylint: disable=invalid-name,missing-function-docstring,relative-beyond-top-level,g-long-lambda
from typing import Any, List, Optional, cast

from ..cmake_builder import CMakeBuilder
from ..cmake_builder import quote_list
from ..cmake_target import CMakeDepsProvider
from ..cmake_target import CMakeTarget
from ..evaluation import EvaluationState
from ..native_rules_proto import get_proto_output_dir
from ..native_rules_proto import PluginSettings
from ..native_rules_proto import PROTO_COMPILER
from ..starlark.bazel_globals import BazelGlobals
from ..starlark.bazel_globals import register_bzl_library
from ..starlark.bazel_target import TargetId
from ..starlark.common_providers import FilesProvider
from ..starlark.common_providers import ProtoLibraryProvider
from ..starlark.invocation_context import InvocationContext
from ..starlark.invocation_context import RelativeLabel
from ..starlark.provider import TargetInfo
from ..starlark.select import Configurable

_SEP = "\n        "
_GRPC = PluginSettings(
    TargetId("@com_github_grpc_grpc//src/compiler:grpc_cpp_plugin"), "grpc",
    [".grpc.pb.h", ".grpc.pb.cc"],
    [TargetId("@com_github_grpc_grpc//:grpc++_codegen_proto")])


@register_bzl_library(
    "@com_github_grpc_grpc//bazel:generate_cc.bzl", build=True)
class GrpcGenerateCcLibrary(BazelGlobals):

  def bazel_generate_cc(self,
                        well_known_protos: Any,
                        name: str,
                        visibility: Optional[List[RelativeLabel]] = None,
                        **kwargs):
    context = self._context.snapshot()
    target = context.resolve_target(name)
    context.add_rule(
        target,
        lambda: _generate_cc_impl(
            context, target, well_known_protos=well_known_protos, **kwargs),
        analyze_by_default=False)


def _generate_cc_impl(_context: InvocationContext,
                      _target: TargetId,
                      srcs: Optional[Configurable[List[RelativeLabel]]] = None,
                      plugin: Optional[Configurable[RelativeLabel]] = None,
                      flags: Optional[List[str]] = None,
                      **kwargs):
  del kwargs
  resolved_srcs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(srcs))
  assert len(resolved_srcs) == 1
  proto_library_target = resolved_srcs[0]

  proto_info = _context.get_target_info(proto_library_target).get(
      ProtoLibraryProvider)
  assert proto_info is not None

  state = _context.access(EvaluationState)
  cmake_deps: List[CMakeTarget] = []
  cmake_target_pair = state.generate_cmake_target_pair(_target, alias=False)
  protoc_deps = CMakeDepsProvider([cmake_target_pair.target])

  plugin_settings = _GRPC
  if plugin is not None:
    resolved_plugin = _context.resolve_target_or_label(
        cast(RelativeLabel, _context.evaluate_configurable(plugin)))
    plugin_settings = PluginSettings(
        resolved_plugin, "grpc", [".grpc.pb.h", ".grpc.pb.cc"],
        [TargetId("@com_github_grpc_grpc//:grpc++_codegen_proto")])

  cmake_deps.extend(state.get_dep(PROTO_COMPILER))

  # Construct the generated paths, installing this rule as a dependency.
  proto_src_files = []
  generated_paths = []
  for src in proto_info.srcs:
    proto_src_files.extend(state.get_file_paths(src, cmake_deps))
    assert src.target_name.endswith(".proto"), f"{repr(src)} must end in .proto"
    proto_prefix = src.target_name[:-len(".proto")]
    for ext in plugin_settings.exts:
      generated_target = src.get_target_id(f"{proto_prefix}{ext}")
      generated_path = _context.get_generated_file_path(generated_target)
      _context.add_analyzed_target(
          generated_target,
          TargetInfo(FilesProvider([generated_path]), protoc_deps))
      generated_paths.append(generated_path)

  plugin_name = state.get_dep(plugin_settings.plugin)
  if len(plugin_name) != 1:
    raise ValueError(
        f"Resolving {plugin_settings.plugin} returned: {plugin_name}")
  cmake_deps.append(plugin_name[0])
  plugin = f"\n    PLUGIN protoc-gen-{plugin_settings.name}=$<TARGET_FILE:{plugin_name[0]}>"

  # Construct the output path. This is also the target include dir.
  # ${PROJECT_BINARY_DIR}
  output_dir = get_proto_output_dir(_context, proto_info.strip_import_prefix)

  import_target = state.generate_cmake_target_pair(resolved_srcs[0]).target
  cmake_name = cmake_target_pair.target

  builder = _context.access(CMakeBuilder)
  builder.addtext(f"""
# {_target.as_label()}
add_custom_target({cmake_name})
target_sources({cmake_name} PRIVATE{_SEP}{quote_list(proto_src_files , separator=_SEP)})

btc_protobuf(
    TARGET {cmake_name}
    IMPORT_TARGETS  {import_target}
    LANGUAGE {plugin_settings.name}{plugin}
    GENERATE_EXTENSIONS {quote_list(plugin_settings.exts)}
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PLUGIN_OPTIONS {quote_list(flags)}
    DEPENDENCIES {quote_list(cmake_deps)}
    PROTOC_OUT_DIR {output_dir}
)
""")

  _context.add_analyzed_target(
      _target,
      TargetInfo(*cmake_target_pair.as_providers(),
                 FilesProvider(generated_paths)))
