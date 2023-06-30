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
"""

# pylint: disable=invalid-name

import pathlib
from typing import Dict, List, NamedTuple, Optional

from .cmake_builder import CMakeBuilder
from .cmake_builder import quote_list
from .cmake_target import CMakeLibraryTargetProvider
from .cmake_target import CMakeTarget
from .emit_cc import emit_cc_library
from .evaluation import EvaluationState
from .starlark.bazel_globals import register_native_build_rule
from .starlark.bazel_target import RepositoryId
from .starlark.bazel_target import TargetId
from .starlark.common_providers import FilesProvider
from .starlark.common_providers import ProtoLibraryProvider
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.provider import TargetInfo


class PluginSettings(NamedTuple):
  name: str
  plugin: Optional[TargetId]
  exts: List[str]
  runtime: List[TargetId]
  replacement_targets: Dict[TargetId, Optional[TargetId]]
  language: Optional[str] = None


PROTO_REPO = RepositoryId("com_google_protobuf")
PROTO_COMPILER = PROTO_REPO.parse_target("//:protoc")
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
    # Descriptor.proto isn't considered "well known", but is available via
    # :protobuf and :protobuf_wkt
    "descriptor",
]


_REPLACEMENTS = dict(
    [
        (
            PROTO_REPO.parse_target(f"//src/google/protobuf:{x}_proto"),
            PROTO_RUNTIME,
        )
        for x in _WELL_KNOWN_TYPES
    ]
    + [
        (
            PROTO_REPO.parse_target(f"//:{x}_proto"),
            PROTO_RUNTIME,
        )
        for x in _WELL_KNOWN_TYPES
    ]
    + [
        (
            PROTO_REPO.parse_target("//src/google/protobuf/compiler:plugin"),
            PROTO_REPO.parse_target(
                "//src/google/protobuf/compiler:code_generator"
            ),
        ),
    ]
)


_CC = PluginSettings(
    name="cpp",
    plugin=None,
    exts=[".pb.h", ".pb.cc"],
    runtime=[PROTO_RUNTIME],
    replacement_targets=_REPLACEMENTS,
)


def _get_proto_output_dir(
    _context: InvocationContext, strip_import_prefix: Optional[str]
) -> str:
  """Construct the output path for the proto compiler.

  This is typically a path relative to ${PROJECT_BINARY_DIR} where the
  protocol compiler will output copied protos.
  """
  output_dir = "${PROJECT_BINARY_DIR}"
  if strip_import_prefix is not None:
    relative_package_path = pathlib.PurePosixPath(
        _context.caller_package_id.package_name
    )
    include_path = str(
        relative_package_path.joinpath(
            pathlib.PurePosixPath(strip_import_prefix)
        )
    )
    if include_path[0] == "/":
      include_path = include_path[1:]
    output_dir = f"${{PROJECT_BINARY_DIR}}/{include_path}"
  return output_dir


def btc_protobuf(
    _context: InvocationContext,
    cmake_name: CMakeTarget,
    proto_library_target: TargetId,
    plugin_settings: PluginSettings,
    cmake_deps: Optional[List[CMakeTarget]] = None,
    flags: Optional[List[str]] = None,
) -> str:
  """Generate text to invoke btc_protobuf for a single target."""
  if not cmake_deps:
    cmake_deps = []

  target_info = _context.get_target_info(proto_library_target)
  proto_cmake_target = target_info.get(CMakeLibraryTargetProvider).target

  proto_info = target_info.get(ProtoLibraryProvider)
  assert proto_info is not None

  state = _context.access(EvaluationState)

  cmake_deps.extend(state.get_dep(PROTO_COMPILER))
  cmake_deps = list(sorted(set(cmake_deps)))

  language = (
      plugin_settings.language
      if plugin_settings.language
      else plugin_settings.name
  )

  plugin = ""
  if plugin_settings.plugin:
    plugin_name = state.get_dep(plugin_settings.plugin)
    if len(plugin_name) != 1:
      raise ValueError(
          f"Resolving {plugin_settings.plugin} returned: {plugin_name}"
      )

    cmake_deps.append(plugin_name[0])
    plugin = (
        f"\n    PLUGIN protoc-gen-{language}=$<TARGET_FILE:{plugin_name[0]}>"
    )

  # Construct the output path. This is also the target include dir.
  # ${PROJECT_BINARY_DIR}
  output_dir = _get_proto_output_dir(_context, proto_info.strip_import_prefix)

  plugin_flags = ""
  if flags:
    plugin_flags = f"\n    PLUGIN_OPTIONS {quote_list(flags)}"

  return f"""
btc_protobuf(
    TARGET {cmake_name}
    PROTO_TARGET {proto_cmake_target}
    LANGUAGE {language}
    GENERATE_EXTENSIONS {quote_list(plugin_settings.exts)}
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR {output_dir}{plugin}{plugin_flags}
    DEPENDENCIES {quote_list(cmake_deps)}
)
"""


def _generate_proto_library_target(
    _context: InvocationContext,
    target: TargetId,
    *,
    plugin: PluginSettings,
    dep_plugin: Optional[PluginSettings],
    output_dep_targets: List[TargetId],
):
  """Emit or return an appropriate TargetId for protos compiled."""
  state = _context.access(EvaluationState)

  # This is a reference to a proto where code-generation has been
  # excluded, so link the replacement target.
  if target in plugin.replacement_targets:
    replacement = plugin.replacement_targets[target]
    if replacement:
      output_dep_targets.append(replacement)
    return

  cc_library_target = target.get_target_id(
      f"{target.target_name}__{plugin.name}_library"
  )

  # The generated code may also be replaced; if so, return that.
  if cc_library_target in plugin.replacement_targets:
    replacement = plugin.replacement_targets[cc_library_target]
    if replacement:
      output_dep_targets.append(replacement)
    return

  # The generated target is always a dependency now.
  output_dep_targets.append(cc_library_target)

  # This library could already have been constructed.
  info = state.get_optional_target_info(cc_library_target)
  if info is not None:
    return

  # First-party proto references must exist.
  if _context.caller_package_id.repository_id == target.repository_id:
    target_info = state.get_target_info(target)
  else:
    target_info = state.get_optional_target_info(target)

  cc_deps: List[CMakeTarget] = []
  cmake_deps: List[CMakeTarget] = state.get_dep(PROTO_COMPILER)
  import_target: Optional[CMakeTarget] = None
  proto_src_files: List[str] = []
  valid_target = False

  # dep_plugin specified; generate it.
  if dep_plugin is not None:
    dep_targets: List[TargetId] = []
    _generate_proto_library_target(
        _context,
        target,
        plugin=dep_plugin,
        dep_plugin=None,
        output_dep_targets=dep_targets,
    )
    cc_deps.extend(state.get_deps(dep_targets))

  # Run genproto on each dependency.
  proto_info = target_info.get(ProtoLibraryProvider) if target_info else None
  if proto_info is not None:
    dep_targets: List[TargetId] = []
    for dep in proto_info.deps:
      _generate_proto_library_target(
          _context,
          dep,
          plugin=plugin,
          dep_plugin=dep_plugin,
          output_dep_targets=dep_targets,
      )

    cc_deps.extend(state.get_deps(dep_targets))

    # NOTE: Consider using generator expressions to add to the library target.
    # Something like  $<TARGET_PROPERTY:target,INTERFACE_SOURCES>
    for src in proto_info.srcs:
      proto_src_files.extend(state.get_file_paths(src, cmake_deps))

    import_target = state.generate_cmake_target_pair(target).target
    valid_target = True

  # ... It may be exported as a FilesProvider instead.
  files_info = (
      target_info.get(FilesProvider)
      if target_info and not valid_target
      else None
  )
  if files_info is not None:
    for src in files_info.paths:
      if src.endswith(".proto"):
        proto_src_files.extend(state.get_file_paths(src, cmake_deps))

    if proto_src_files:
      import_target = state.generate_cmake_target_pair(target).target
      valid_target = True

  provider = (
      target_info.get(CMakeLibraryTargetProvider)
      if target_info and not valid_target
      else None
  )
  if provider:
    import_target = provider.target
    valid_target = True

  if not valid_target:
    # This target is not available; construct an ephemeral reference.
    print(
        f"Blind reference to {cc_library_target.as_label()} for"
        f" {target.as_label()} from {_context.caller_package_id}"
    )
    return

  # Get our cmake name; proto libraries need aliases to be referenced
  # from other source trees.
  cmake_target_pair = state.generate_cmake_target_pair(cc_library_target)
  proto_src_files = sorted(set(proto_src_files))

  if not proto_src_files and not cc_deps and not import_target:
    raise ValueError(
        f"Proto generation failed: {target.as_label()} no inputs for"
        f" {cc_library_target.as_label()}"
    )

  for dep in plugin.runtime:
    cc_deps.extend(state.get_dep(dep))

  btc_protobuf_txt = ""
  if proto_src_files:
    btc_protobuf_txt = btc_protobuf(
        _context,
        cmake_target_pair.target,
        target,
        plugin,
        cmake_deps,
    )

  builder = _context.access(CMakeBuilder)
  builder.addtext(f"\n# {cc_library_target.as_label()}")
  emit_cc_library(
      builder,
      cmake_target_pair,
      hdrs=set(),
      srcs=set(),
      deps=set(cc_deps),
      header_only=(not bool(proto_src_files)),
      includes=[],
  )
  builder.addtext(btc_protobuf_txt)

  _context.add_analyzed_target(
      cc_library_target, TargetInfo(*cmake_target_pair.as_providers())
  )


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
          context, target, _CC, "cc_proto_library", **kwargs
      ),
      visibility=visibility,
  )


def cc_proto_library_impl(
    _context: InvocationContext,
    _target: TargetId,
    _plugin: PluginSettings,
    _mnemonic: str,
    *,
    _dep_plugin: Optional[PluginSettings] = None,
    _extra_deps: Optional[List[RelativeLabel]] = None,
    deps: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  del kwargs
  resolved_deps = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(deps)
  )

  state = _context.access(EvaluationState)
  cmake_target_pair = state.generate_cmake_target_pair(_target)

  # Add additional deps, if requested.
  library_deps: List[CMakeTarget] = []
  if _extra_deps:
    resolved_extra_deps = _context.resolve_target_or_label_list(
        _context.evaluate_configurable_list(_extra_deps)
    )
    library_deps.extend(state.get_deps(resolved_extra_deps))

  # Typically there is a single proto dep in a cc_library_target, multiple are
  # supported, thus we resolve each library target here.
  for dep_target in resolved_deps:
    dep_targets: List[TargetId] = []
    _generate_proto_library_target(
        _context,
        dep_target,
        plugin=_plugin,
        dep_plugin=_dep_plugin,
        output_dep_targets=dep_targets,
    )
    library_deps.extend(state.get_deps(dep_targets))

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
