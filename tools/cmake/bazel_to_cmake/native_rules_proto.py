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
target (e.g. cc_proto_library) just links those targets. Cross-repository
dependencies can link the per-source targets as well.
"""

# pylint: disable=invalid-name

import io
import os
import pathlib
from typing import List, NamedTuple, Optional, Set

from .cmake_builder import CMakeBuilder
from .cmake_builder import quote_list
from .cmake_builder import quote_path
from .cmake_target import CMakeDepsProvider
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


PROTO_SUFFIX = ".proto"

_PROTO_COMPILER = TargetId("@com_google_protobuf//:protoc")

_WELL_KNOWN_PROTOS: Set[TargetId] = set([
    TargetId("@com_google_protobuf//:any_proto"),
    TargetId("@com_google_protobuf//:api_proto"),
    TargetId("@com_google_protobuf//:compiler_plugin_proto"),
    TargetId("@com_google_protobuf//:descriptor_proto"),
    TargetId("@com_google_protobuf//:duration_proto"),
    TargetId("@com_google_protobuf//:empty_proto"),
    TargetId("@com_google_protobuf//:field_mask_proto"),
    TargetId("@com_google_protobuf//:source_context_proto"),
    TargetId("@com_google_protobuf//:struct_proto"),
    TargetId("@com_google_protobuf//:timestamp_proto"),
    TargetId("@com_google_protobuf//:type_proto"),
    TargetId("@com_google_protobuf//:wrappers_proto"),
])

_WELL_KNOWN_PROTO_TARGETS = {
    "cpp": TargetId("@com_google_protobuf//:protobuf"),
    "upb": TargetId("@com_google_protobuf//:well_known_protos_ubp"),
    "upbdefs": TargetId("@com_google_protobuf//:well_known_protos_ubpdefs"),
}

_CC = PluginSettings(None, "cpp", [".pb.h", ".pb.cc"],
                     [TargetId("@com_google_protobuf//:protobuf")])


class ProtocOutputTuple(NamedTuple):
  generated_target: TargetId
  cmake_target: CMakeTarget
  files: FilesProvider


def get_proto_plugin_library_target(_context: InvocationContext, *,
                                    plugin_settings: PluginSettings,
                                    target: TargetId) -> TargetId:
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

    # otherwise fallback to the proto library target.
    target_info = state.get_optional_target_info(target)
    if target_info is None:
      return cc_library_target
    proto_info = target_info.get(ProtoLibraryProvider)
    if proto_info is None:
      print(f"{_context.caller_package_id} references {target}")
      return cc_library_target

  # Library target not found; run the protoc compiler and build the library
  # target here.
  custom_target_deps = []
  hdrs: Set[str] = set()
  srcs: Set[str] = set()

  for src in proto_info.srcs:
    generated = protoc_compile_protos_impl(
        _context,
        plugin_settings=plugin_settings,
        proto_src=src,
        proto_library_target=target,
        strip_import_prefix=proto_info.strip_import_prefix)
    custom_target_deps.append(generated.cmake_target)
    hdrs.update(filter(lambda x: x.endswith(".h"), generated.files.paths))
    srcs.update(filter(lambda x: not x.endswith(".h"), generated.files.paths))

  cc_deps: List[TargetId] = [
      get_proto_plugin_library_target(
          _context, plugin_settings=plugin_settings, target=dep)
      for dep in proto_info.deps
  ]
  cc_deps.extend(plugin_settings.deps)

  cmake_cc_target_pair = state.generate_cmake_target_pair(
      cc_library_target, generate_alias=True)
  _context.add_analyzed_target(cc_library_target,
                               TargetInfo(*cmake_cc_target_pair.as_providers()))

  builder = _context.access(CMakeBuilder)
  emit_cc_library(
      builder,
      cmake_cc_target_pair,
      hdrs=hdrs,
      srcs=srcs,
      deps=set(state.get_deps(cc_deps)),
      custom_target_deps=custom_target_deps,
  )
  for x in custom_target_deps:
    builder.addtext(
        f"""target_include_directories({cmake_cc_target_pair.target} PUBLIC
         $<BUILD_INTERFACE:$<TARGET_PROPERTY:{x},INTERFACE_INCLUDE_DIRECTORIES>>)\n"""
    )

  return cc_library_target


def protoc_compile_protos_impl(
    _context: InvocationContext,
    *,
    plugin_settings: PluginSettings,
    proto_library_target: TargetId,
    proto_src: TargetId,
    target: Optional[TargetId] = None,
    flags: Optional[List[str]] = None,
    strip_import_prefix: Optional[str] = None) -> ProtocOutputTuple:
  if flags is None:
    flags = []

  if target is None:
    target = proto_src.get_target_id(
        f"{proto_src.target_name}__{plugin_settings.name}_protoc")

  state = _context.access(EvaluationState)

  # Construct expected generated file paths.
  assert proto_src.target_name.endswith(
      PROTO_SUFFIX), f"{proto_src} must end in {PROTO_SUFFIX}"
  proto_prefix = proto_src.target_name[:-len(PROTO_SUFFIX)]

  generated_filenames = [
      proto_src.get_target_id(f"{proto_prefix}{ext}")
      for ext in plugin_settings.exts
  ]
  generated_paths = [
      _context.get_generated_file_path(f) for f in generated_filenames
  ]

  # Construct the output path. This is also the target include dir.
  # ${PROJECT_BINARY_DIR}
  if strip_import_prefix:
    include_dir = str(
        pathlib.PurePosixPath(
            _context.caller_package.repository.cmake_binary_dir).joinpath(
                strip_import_prefix))
  else:
    include_dir = str(
        pathlib.PurePosixPath(
            _context.caller_package.repository.cmake_binary_dir))

  # Construct protoc args. plugin-naming is special; protoc expects it
  # to be protoc-gen-<something>, and then uses <something>_out as the
  # output parameter.
  cmake_deps: List[CMakeTarget] = []
  extra_args: List[str] = []

  if plugin_settings.plugin is not None:
    cmake_name = state.get_dep(plugin_settings.plugin)
    if len(cmake_name) != 1:
      raise ValueError(
          f"Resolving {plugin_settings.plugin} returned: {cmake_name}")

    cmake_deps.append(cmake_name[0])
    extra_args.append(
        f"--plugin=protoc-gen-{plugin_settings.name}=$<TARGET_FILE:{cmake_name[0]}>"
    )

  joined_flags = ",".join(flags)
  extra_args.append(
      f"--{plugin_settings.name}_out={joined_flags}:{include_dir}")

  # Add Providers and generated file targets to the context.
  protoc_target_pair = state.generate_cmake_target_pair(
      target, generate_alias=False)

  protoc_deps = CMakeDepsProvider([protoc_target_pair.target])

  for a, b in zip(generated_filenames, generated_paths):
    _context.add_analyzed_target(a, TargetInfo(FilesProvider([b]), protoc_deps))

  files_provider = FilesProvider(generated_paths)
  _context.add_analyzed_target(
      target, TargetInfo(*protoc_target_pair.as_providers(), files_provider))

  # Emit the builder.
  proto_src_files = state.get_file_paths(proto_src, cmake_deps)
  assert len(proto_src_files) == 1

  # Resolve the protoc compiler.
  protoc_name = state.get_dep(_PROTO_COMPILER)[0]
  cmake_deps.append(protoc_name)

  # Resolve the dependencies library
  cmake_include_target = state.generate_cmake_target_pair(
      proto_library_target, generate_alias=False).target

  _emit_protoc_generate(
      _context.access(CMakeBuilder),
      protoc_name,
      cmake_target=protoc_target_pair.target,
      cmake_include_target=cmake_include_target,
      proto_file_path=CMakeTarget(proto_src_files[0]),
      generated=generated_paths,
      cmake_deps=cmake_deps,
      extra_args=extra_args,
      include_dir=include_dir,
      comment=f"Running protoc ({plugin_settings.name})")

  return ProtocOutputTuple(target, protoc_target_pair.target, files_provider)


def _emit_protoc_generate(
    _builder: CMakeBuilder,
    protoc_cmake_target: CMakeTarget,
    *,
    cmake_target: CMakeTarget,
    cmake_include_target: CMakeTarget,
    proto_file_path: CMakeTarget,
    generated: List[str],
    cmake_deps: List[CMakeTarget],
    include_dir: str,
    extra_args: Optional[List[str]] = None,
    comment="Running protoc",
):
  """Emits CMake to generates a C++ file from a Proto file using protoc."""
  if extra_args is None:
    extra_args = []
  # The cmake_include_target is used to get the recursive
  # INTERFACE_INCLUDE_DIRECTORIES property declared by the proto_library() rule.
  assert cmake_include_target
  cmake_deps.append(proto_file_path)
  cmake_deps.append(cmake_include_target)

  out = io.StringIO()
  out.write(f"""
add_custom_command(
  OUTPUT {quote_list(generated)}
  COMMAND {protoc_cmake_target}
  ARGS --experimental_allow_proto3_optional""")
  for arg in extra_args:
    out.write(f"\n       {arg}")
  out.write(f"""
      "-I$<JOIN:$<TARGET_PROPERTY:{cmake_include_target},INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
      {quote_path(proto_file_path)}
  DEPENDS {quote_list(sorted(set(cmake_deps)))}
  COMMENT "{comment} on {os.path.basename(proto_file_path)}"
  COMMAND_EXPAND_LISTS
  VERBATIM)
add_custom_target({cmake_target} DEPENDS {quote_list(generated)})
set_target_properties({cmake_target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES {include_dir})
""")
  _builder.addtext(out.getvalue())


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

  cmake_name = state.generate_cmake_target_pair(
      _target, generate_alias=False).target

  # Validate src properties: files ending in .proto within the same repo.
  for t in resolved_srcs:
    assert t.target_name.endswith(
        PROTO_SUFFIX), f"{t} must end in {PROTO_SUFFIX}"
    # Verify that the source is in the same repository as the proto_library rule
    assert t.repository_id == _target.repository_id

  # Resolve deps
  cmake_proto_deps = []
  for d in resolved_deps:
    state.get_optional_target_info(d)
    cmake_proto_deps.append(
        state.generate_cmake_target_pair(d, generate_alias=False).target)

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

  value = " ".join(includes)
  includes_name = cmake_name + "_INCLUDES"
  includes_literal = "${" + includes_name + "}"

  out = io.StringIO()
  out.write(f"""
add_custom_target({cmake_name})
list(APPEND {includes_name} {value})
""")
  if cmake_proto_deps:
    include_deps = " ".join(sorted(set(cmake_proto_deps)))
    out.write(f"""
foreach(t {include_deps})
get_property(_target_includes TARGET "${{t}}" PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
list(APPEND {includes_name} ${{_target_includes}})
unset(_target_includes)
endforeach()
list(REMOVE_DUPLICATES {includes_name})
""")

  out.write(f"""
set_target_properties({cmake_name} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "{includes_literal}")
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
  cmake_target_pair = state.generate_cmake_target_pair(
      _target, generate_alias=True)

  # Typically there is a single proto dep in a cc_library_target, multiple are
  # supported, thus we resolve each library target here.
  dep_library_targets = [
      get_proto_plugin_library_target(
          _context, plugin_settings=_CC, target=dep_target)
      for dep_target in resolved_deps
  ]
  emit_cc_library(
      _context.access(CMakeBuilder),
      cmake_target_pair,
      hdrs=set(),
      srcs=set(),
      deps=set(state.get_deps(dep_library_targets)),
  )
  _context.add_analyzed_target(_target,
                               TargetInfo(*cmake_target_pair.as_providers()))
