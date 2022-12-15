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
"""Functions to assist in invoking protoc for CMake.

proto_library and friends are somewhat special in the rules department. Normally
Bazel has a global view of the dependency tree, and can apply aspects() and
other mechanisms to ensure that the proper code is generated.

However, bazel_to_cmake does not have a global view; each repository is
processed independently. In light of this, all proto dependency targets are
expected to be able to link to targets which have standard rule names, and
bazel_to_cmake should inject code to build those targets when missing.

To achieve this, cc_proto_library, grpc_proto_library, etc. build protoc
plugin-specific proto library targets for each source, and the top-level
(public) target (e.g. cc_proto_library) just links those targets.
Cross-repository dependencies can link the per-source targets as well.

This generator assumes that appropriate mappings between bazel and CMake targets
have been configured:
  @com_google_protobuf//:protobuf      => protobuf::libprotobuf
  @com_google_protobuf//:protobuf_lite => protobuf::libprotobuf-lite
  @com_google_protobuf//:protoc        => protobuf::protoc
  @com_google_protobuf//:protoc_lib    => protobuf::libprotoc

In debian-like systems, use of system versions requires the following packages:
  apt install libprotobuf-dev libprotoc-dev protobuf-compiler
"""

# pylint: disable=invalid-name

import io
import pathlib
from typing import List, NamedTuple, Optional, Set

from .cmake_builder import CMakeBuilder
from .cmake_builder import quote_list
from .cmake_target import CMakeTarget
from .emit_cc import emit_cc_library
from .evaluation import EvaluationState
from .starlark.bazel_globals import register_native_build_rule
from .starlark.bazel_target import TargetId
from .starlark.common_providers import FilesProvider
from .starlark.common_providers import ProtoLibraryProvider
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.provider import TargetInfo


class PluginSettings(NamedTuple):
  plugin: Optional[TargetId]
  name: str
  exts: List[str]
  deps: List[TargetId]


PROTO_COMPILER = TargetId("@com_google_protobuf//:protoc")
_SEP = "\n        "

_CC = PluginSettings(None, "cpp", [".pb.h", ".pb.cc"],
                     [TargetId("@com_google_protobuf//:protobuf")])

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
    # Descriptor.proto isn't considered "well known", but is available via
    # :protobuf and :protobuf_wkt
    "descriptor"
]

_WELL_KNOWN_PROTOS: Set[TargetId] = set(
    [TargetId(f"@com_google_protobuf//:{x}_proto") for x in _WELL_KNOWN_TYPES] +
    [
        TargetId(f"@com_google_protobuf//src/google/protobuf:{x}_proto")
        for x in _WELL_KNOWN_TYPES
    ])

_WELL_KNOWN_PROTO_TARGETS = {
    "cpp":
        TargetId("@com_google_protobuf//:protobuf"),  # wkt_cc_proto
    "upb":
        TargetId("@local_proto_mirror//google/protobuf:well_known_protos_upb"),
    "upbdefs":
        TargetId(
            "@local_proto_mirror//google/protobuf:well_known_protos_upbdefs"),
}

_OTHER_CORE_PROTOS = {
    ("cpp",
     TargetId("@com_google_protobuf//src/google/protobuf/compiler:plugin")):
        TargetId(
            "@com_google_protobuf//src/google/protobuf/compiler:code_generator")
}


class ProtocOutputTuple(NamedTuple):
  generated_target: TargetId
  cmake_target: CMakeTarget
  files: FilesProvider


def get_proto_output_dir(_context: InvocationContext,
                         strip_import_prefix: Optional[str]) -> str:
  """Construct the output path for the proto compiler.

  This is typically a path relative to ${PROJECT_BINARY_DIR} where the
  protocol compiler will output copied protos.
  """
  output_dir = "${PROJECT_BINARY_DIR}"
  if strip_import_prefix is not None:
    relative_package_path = pathlib.PurePosixPath(
        _context.caller_package_id.package_name)
    include_path = str(
        relative_package_path.joinpath(
            pathlib.PurePosixPath(strip_import_prefix)))
    if include_path[0] == "/":
      include_path = include_path[1:]
    output_dir = f"${{PROJECT_BINARY_DIR}}/{include_path}"
  return output_dir


def get_proto_plugin_library_target(_context: InvocationContext, *,
                                    plugin_settings: PluginSettings,
                                    target: TargetId) -> TargetId:
  """Emit or return an appropriate TargetId for protos compiled."""
  cc_library_target = target.get_target_id(
      f"{target.target_name}__{plugin_settings.name}_library")

  state = _context.access(EvaluationState)

  # This library could already have been constructed.
  info = state.get_optional_target_info(cc_library_target)
  if info is not None:
    return cc_library_target

  if _context.caller_package_id.repository_id == target.repository_id:
    # This is a first-party repository, and should be available.
    proto_info = state.get_target_info(target).get(ProtoLibraryProvider)
  else:
    # This is a reference to a non-first-party repository,
    # which merits special treatment. First look for the WELL_KNOWN_PROTO
    # targets, which have special link targets.
    if target in _WELL_KNOWN_PROTOS:
      if plugin_settings.name in _WELL_KNOWN_PROTO_TARGETS:
        return _WELL_KNOWN_PROTO_TARGETS[plugin_settings.name]

    # Then look at other replacment targets.
    replacement = _OTHER_CORE_PROTOS.get((plugin_settings.name, target))
    if replacement is not None:
      return replacement

    # otherwise fallback to the proto library target.
    target_info = state.get_optional_target_info(target)
    if target_info is None:
      return cc_library_target
    proto_info = target_info.get(ProtoLibraryProvider)
    if proto_info is None:
      print(f"{_context.caller_package_id} references {target}")
      return cc_library_target

  # Get our cmake name; note that proto libraries do not have aliases.
  cmake_target_pair = state.generate_cmake_target_pair(
      cc_library_target, alias=False)

  # Library target not found; run the protoc compiler and build the library
  # target here.
  cc_deps: List[TargetId] = [
      get_proto_plugin_library_target(
          _context, plugin_settings=plugin_settings, target=dep)
      for dep in proto_info.deps
  ]
  cc_deps.extend(plugin_settings.deps)

  # NOTE: Consider using generator expressions to add to the library target.
  # Something like  $<TARGET_PROPERTY:target,INTERFACE_SOURCES>
  cmake_deps: List[CMakeTarget] = []
  proto_src_files = []
  for src in proto_info.srcs:
    proto_src_files.extend(state.get_file_paths(src, cmake_deps))
  proto_src_files = sorted(set(proto_src_files))

  cmake_deps.extend(state.get_dep(PROTO_COMPILER))

  # Construct the output path. This is also the target include dir.
  # ${PROJECT_BINARY_DIR}
  output_dir = get_proto_output_dir(_context, proto_info.strip_import_prefix)

  plugin = ""
  if plugin_settings.plugin:
    cmake_name = state.get_dep(plugin_settings.plugin)
    if len(cmake_name) != 1:
      raise ValueError(
          f"Resolving {plugin_settings.plugin} returned: {cmake_name}")

    cmake_deps.append(cmake_name[0])
    plugin = f"    PLUGIN protoc-gen-{plugin_settings.name}=$<TARGET_FILE:{cmake_name[0]}>\n"

  import_target = state.generate_cmake_target_pair(target).target

  builder = _context.access(CMakeBuilder)
  builder.addtext(f"\n# {cc_library_target.as_label()}")
  emit_cc_library(
      builder,
      cmake_target_pair,
      hdrs=set(),
      srcs=set(proto_src_files),
      deps=set(state.get_deps(cc_deps)),
  )

  dependencies = ""
  if cmake_deps:
    dependencies = f"    DEPENDENCIES {quote_list(cmake_deps)}\n"

  # PLUGIN_OPTIONS {quote_list(flags)}
  builder.addtext(f"""
btc_protobuf(
    TARGET {cmake_target_pair.target}
    IMPORT_TARGETS  {import_target}
    LANGUAGE {plugin_settings.name}
    GENERATE_EXTENSIONS {quote_list(plugin_settings.exts)}
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR {output_dir}
{plugin}{dependencies})
""")

  _context.add_analyzed_target(cc_library_target,
                               TargetInfo(*cmake_target_pair.as_providers()))
  return cc_library_target


@register_native_build_rule
def proto_library(self: InvocationContext,
                  name: str,
                  visibility: Optional[List[RelativeLabel]] = None,
                  **kwargs):
  context = self.snapshot()
  target = context.resolve_target(name)
  context.add_rule(
      target,
      lambda: _proto_library_impl(context, target, **kwargs),
      visibility=visibility)


def _proto_library_impl(_context: InvocationContext,
                        _target: TargetId,
                        srcs: Optional[List[RelativeLabel]] = None,
                        deps: Optional[List[RelativeLabel]] = None,
                        strip_import_prefix: Optional[str] = None,
                        **kwargs):
  del kwargs
  resolved_srcs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(srcs))
  resolved_deps = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(deps))

  state = _context.access(EvaluationState)

  cmake_name = state.generate_cmake_target_pair(_target, alias=False).target

  # Validate src properties: files ending in .proto within the same repo,
  # and add them to the proto_src_files.
  cmake_deps: List[CMakeTarget] = []
  proto_src_files = []
  for proto in resolved_srcs:
    assert proto.target_name.endswith(".proto"), f"{proto} must end in .proto"
    # Verify that the source is in the same repository as the proto_library rule
    assert proto.repository_id == _target.repository_id
    proto_src_files.extend(state.get_file_paths(proto, cmake_deps))

  # Resolve deps. When using system protobuffers, well-known-proto targets need
  # 'Protobuf_IMPORT_DIRS' added to their transitive includes.
  import_vars = ""
  import_targets = ""
  for d in resolved_deps:
    if d in _WELL_KNOWN_PROTOS:
      import_vars = "Protobuf_IMPORT_DIRS"
    state.get_optional_target_info(d)
    import_targets += f"{state.generate_cmake_target_pair(d).target} "

  # In order to propagate the includes to our compile targets, each
  # proto_library() becomes a custom CMake target which contains an
  # INTERFACE_INCLUDE_DIRECTORIES property which can be used by the protoc
  # compiler.
  repo = state.workspace.repos.get(_target.repository_id)
  assert repo is not None
  source_dir = repo.source_directory
  bin_dir = repo.cmake_binary_dir
  if strip_import_prefix:
    source_dir = str(
        pathlib.PurePosixPath(source_dir).joinpath(strip_import_prefix))
    bin_dir = str(pathlib.PurePosixPath(bin_dir).joinpath(strip_import_prefix))
  includes = set()
  for path in state.get_targets_file_paths(resolved_srcs):
    if path.startswith(source_dir):
      includes.add(source_dir)
    if path.startswith(bin_dir):
      includes.add(bin_dir)

  includes_name = f"{cmake_name}_IMPORT_DIRS"
  includes_literal = "${" + includes_name + "}"
  quoted_srcs = quote_list(sorted(set(proto_src_files)), separator=_SEP)

  out = io.StringIO()
  out.write(f"""
# {_target.as_label()}
add_library({cmake_name} INTERFACE)
target_sources({cmake_name} INTERFACE{_SEP}{quoted_srcs})""")
  if import_vars or import_targets:
    out.write(f"""
btc_transitive_import_dirs(
    OUT_VAR {includes_name}
    IMPORT_DIRS {quote_list(includes)}
    IMPORT_TARGETS {import_targets}
    IMPORT_VARS {import_vars}
)""")
  else:
    out.write(f"\nlist(APPEND {includes_name} {quote_list(includes)})")
  out.write(f"""
set_property(TARGET {cmake_name} PROPERTY INTERFACE_INCLUDE_DIRECTORIES {includes_literal})
""")

  _context.access(CMakeBuilder).addtext(out.getvalue())

  _context.add_analyzed_target(
      _target,
      TargetInfo(
          ProtoLibraryProvider(resolved_srcs, resolved_deps,
                               strip_import_prefix)))


@register_native_build_rule
def cc_proto_library(self: InvocationContext,
                     name: str,
                     visibility: Optional[List[RelativeLabel]] = None,
                     **kwargs):
  context = self.snapshot()
  target = context.resolve_target(name)
  context.add_rule(
      target,
      lambda: _cc_proto_library_impl(context, target, **kwargs),
      visibility=visibility)


def _cc_proto_library_impl(_context: InvocationContext,
                           _target: TargetId,
                           deps: Optional[List[RelativeLabel]] = None,
                           **kwargs):
  del kwargs
  resolved_deps = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(deps))

  state = _context.access(EvaluationState)
  cmake_target_pair = state.generate_cmake_target_pair(_target)

  # Typically there is a single proto dep in a cc_library_target, multiple are
  # supported, thus we resolve each library target here.
  library_deps: List[CMakeTarget] = []
  for dep_target in resolved_deps:
    lib_target = get_proto_plugin_library_target(
        _context, plugin_settings=_CC, target=dep_target)
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
