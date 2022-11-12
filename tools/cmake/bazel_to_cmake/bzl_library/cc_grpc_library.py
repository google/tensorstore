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

from typing import List, Optional, cast

from .. import cmake_builder
from ..evaluation import BazelGlobals
from ..evaluation import EvaluationContext
from ..evaluation import Package
from ..evaluation import register_bzl_library
from ..label import Label
from ..label import RelativeLabel
from ..native_rules_cc import emit_cc_library
from ..provider import CMakeDepsProvider
from ..provider import FilesProvider
from ..provider import ProtoLibraryProvider
from ..provider import TargetInfo


@register_bzl_library(
    "@com_google_tensorstore//bazel:cc_grpc_library.bzl", build=True)
class CcGrpcLibrary(BazelGlobals):

  def bazel_cc_grpc_library(self,
                            name: str,
                            visibility: Optional[List[RelativeLabel]] = None,
                            **kwargs):
    _context = self._context
    package = _context.current_package
    assert package is not None
    label = package.get_label(name)
    _context.add_rule(
        label,
        lambda: _cc_grpc_library_impl(cast(Package, package), label, **kwargs),
        analyze_by_default=package.analyze_by_default(visibility))


def _cc_grpc_library_impl(_package: Package,
                          _label: Label,
                          srcs: Optional[List[RelativeLabel]] = None,
                          deps: Optional[List[RelativeLabel]] = None,
                          **kwargs):
  del kwargs
  if not deps:
    deps = []

  # See: https://github.com/grpc/grpc/blob/master/bazel/cc_grpc_library.bzl
  # Requires:
  #   len(src) == 1. src is a label which references a proto_library
  #     Currently restricted to a single .proto file.
  # Outputs:
  #   gRPC codegen (label + __grpc_codegen) using the protoc compiler.
  #   - plugin: gRPC::grpc_cpp_plugin
  #             (@com_github_grpc_grpc//src/compiler:grpc_cpp_plugin)
  #   - emits .grpc.pb.h, .grpc.pb.cc
  #   cc_library (label)

  _context = _package.context

  grpc_codegen_target = _label + "__grpc_codegen"
  _context.builder.find_package("gRPC")

  assert len(srcs) == 1
  info = _context.get_optional_target_info(_package.get_label(
      srcs[0])).get(ProtoLibraryProvider)

  assert info is not None
  assert len(info.srcs) == 1

  proto_src = next(iter(info.srcs))
  proto_suffix = ".proto"
  assert proto_src.endswith(proto_suffix)

  proto_prefix = proto_src[:-len(proto_suffix)]
  generated_h = f"{proto_prefix}.grpc.pb.h"
  generated_cc = f"{proto_prefix}.grpc.pb.cc"
  generated_h_path = _context.get_generated_file_path(generated_h)
  generated_cc_path = _context.get_generated_file_path(generated_cc)

  # Emit grpc codegen
  generated_target_pair = _context.generate_cmake_target_pair(
      grpc_codegen_target, generate_alias=False)
  generated_deps = CMakeDepsProvider([generated_target_pair.target])
  _context.add_analyzed_target(
      generated_h,
      TargetInfo(FilesProvider([generated_h_path]), generated_deps))
  _context.add_analyzed_target(
      generated_cc,
      TargetInfo(FilesProvider([generated_cc_path]), generated_deps))

  proto_src_files = _context.get_file_paths(proto_src, [])
  assert len(proto_src_files) == 1

  _emit_cc_grpc_library_generate(
      _context.builder,
      generated_target_pair.target,
      proto_src=proto_src_files[0],
      generated_grpc_pb_h=generated_h_path,
      generated_grpc_pb_cc=generated_cc_path,
      cmake_deps=[])

  # Emit the cc_library
  library_target_pair = _context.generate_cmake_target_pair(
      _label, generate_alias=True)

  emit_cc_library(
      _context.builder,
      library_target_pair,
      hdrs=set([generated_h_path]),
      srcs=set([generated_cc_path]),
      deps=_get_grpc_deps(_context, deps),
      custom_target_deps=[generated_target_pair.target])
  _context.add_analyzed_target(_label,
                               TargetInfo(*library_target_pair.as_providers()))


def _get_grpc_deps(_context: EvaluationContext,
                   deps: Optional[List[RelativeLabel]] = None):
  if not deps:
    deps = []

  result = ["@com_github_grpc_grpc//:grpc++_codegen_proto"]
  for dep in deps:
    proto_info = _context.get_target_info(proto_target).get(
        ProtoLibraryProvider)
    if proto_info is None:
      # This could be external; defer to get_deps.
      result.append(dep)
      continue
    # Assume that __cc_proto libraries exist for all sources.
    for proto_lib in proto_info.srcs:
      result.append(proto_lib + "__cc_proto")

  return _context.get_deps(result)


def _emit_cc_grpc_library_generate(
    _builder: cmake_builder.CMakeBuilder,
    cmake_target: str,
    proto_src: str,
    generated_grpc_pb_h: str,
    generated_grpc_pb_cc: str,
    cmake_deps: List[str],
):
  """Generates a C++ grpc file corresponding to a Protobuf."""
  cmake_deps.append("protobuf::protoc")
  cmake_deps.append("gRPC::grpc_cpp_plugin")
  cmake_deps.append(proto_src)
  _builder.addtext(f"""
add_custom_command(
  OUTPUT {cmake_builder.quote_list([generated_grpc_pb_h, generated_grpc_pb_cc])}
  COMMAND protobuf::protoc
  ARGS --experimental_allow_proto3_optional
      --grpc_out "${{PROJECT_BINARY_DIR}}"
      -I "${{PROJECT_SOURCE_DIR}}"
      --plugin=protoc-gen-grpc="$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
      {cmake_builder.quote_path(proto_src)}
  DEPENDS {cmake_builder.quote_list(cmake_deps)}
  COMMENT "Running gRPC compiler on {proto_src}"
  VERBATIM)
add_custom_target({cmake_target} DEPENDS {cmake_builder.quote_list([generated_grpc_pb_h, generated_grpc_pb_cc])})
""")
