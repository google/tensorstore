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

Note that when using system protobuf, the well_known_proto_types are available
via protobuf::libprotobuf. For reference, see

Bazel rules related to protobuf for reference:
https://github.com/bazelbuild/bazel/tree/master/src/main/starlark/builtins_bzl/common/proto/proto_library.bzl
https://github.com/bazelbuild/bazel/tree/master/src/main/starlark/builtins_bzl/common/cc/cc_proto_library.bzl
https://github.com/bazelbuild/rules_proto/tree/master/proto
"""

# pylint: disable=invalid-name

import io
import pathlib
from typing import Dict, List, NamedTuple, Optional, Set

from .cmake_builder import CMakeBuilder
from .cmake_builder import quote_list
from .cmake_builder import quote_path_list
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetProvider
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
from .util import is_relative_to


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

_SEP = "\n        "

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

PROTO_REPLACEMENT_TARGETS: Set[TargetId] = set(
    [
        PROTO_REPO.parse_target(f"//src/google/protobuf:{x}_proto")
        for x in _WELL_KNOWN_TYPES
    ]
    + [PROTO_REPO.parse_target(f"//:{x}_proto") for x in _WELL_KNOWN_TYPES]
)


_CC = PluginSettings(
    name="cpp",
    plugin=None,
    exts=[".pb.h", ".pb.cc"],
    runtime=[PROTO_RUNTIME],
    replacement_targets=dict(
        [(k, PROTO_RUNTIME) for k in PROTO_REPLACEMENT_TARGETS]
        + [(
            PROTO_REPO.parse_target("//src/google/protobuf/compiler:plugin"),
            PROTO_REPO.parse_target(
                "//src/google/protobuf/compiler:code_generator"
            ),
        )]
    ),
)


class ProtocOutputTuple(NamedTuple):
  generated_target: TargetId
  cmake_target: CMakeTarget
  files: FilesProvider


def get_proto_output_dir(
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


def generate_proto_library_target(
    _context: InvocationContext,
    *,
    plugin_settings: PluginSettings,
    target: TargetId,
) -> Optional[TargetId]:
  """Emit or return an appropriate TargetId for protos compiled."""
  state = _context.access(EvaluationState)

  # This is a reference to a proto where code-generation has been
  # excluded, so link the replacement target.
  if target in plugin_settings.replacement_targets:
    return plugin_settings.replacement_targets[target]

  cc_library_target = target.get_target_id(
      f"{target.target_name}__{plugin_settings.name}_library"
  )

  # The generated code may also be replaced; if so, return that.
  if cc_library_target in plugin_settings.replacement_targets:
    return plugin_settings.replacement_targets[cc_library_target]

  # This library could already have been constructed.
  info = state.get_optional_target_info(cc_library_target)
  if info is not None:
    return cc_library_target

  # First-party proto references must exist.
  if _context.caller_package_id.repository_id == target.repository_id:
    target_info = state.get_target_info(target)
  else:
    target_info = state.get_optional_target_info(target)

  if not target_info:
    # This target is not available; construct an ephemeral reference.
    print(
        f"Blind reference to {target.as_label()} from"
        f" {_context.caller_package_id}"
    )
    return cc_library_target

  # Library target not found; genproto on each dependency.
  cc_deps: List[CMakeTarget] = []
  import_target: Optional[CMakeTarget] = None
  cmake_deps: List[CMakeTarget] = state.get_dep(PROTO_COMPILER)
  proto_src_files: List[str] = []

  done = False
  proto_info = target_info.get(ProtoLibraryProvider)

  if proto_info is not None:
    sub_targets: List[TargetId] = []
    for dep in proto_info.deps:
      sub_target_id = generate_proto_library_target(
          _context, plugin_settings=plugin_settings, target=dep
      )
      if sub_target_id:
        sub_targets.append(sub_target_id)
    cc_deps.extend(state.get_deps(list(set(sub_targets))))

    # NOTE: Consider using generator expressions to add to the library target.
    # Something like  $<TARGET_PROPERTY:target,INTERFACE_SOURCES>
    for src in proto_info.srcs:
      proto_src_files.extend(state.get_file_paths(src, cmake_deps))

    import_target = state.generate_cmake_target_pair(target).target
    done = True

  # TODO: Maybe handle FilesProvider like ProtoInfoProvider?

  provider = target_info.get(CMakeTargetProvider)
  if not done and provider:
    import_target = provider.target
    done = True

  if not done:
    print(
        f"Assumed reference to {target.as_label()} from"
        f" {_context.caller_package_id}"
    )
    return cc_library_target

  # Get our cmake name; proto libraries need aliases to be referenced
  # from other source trees.
  cmake_target_pair = state.generate_cmake_target_pair(cc_library_target)
  proto_src_files = sorted(set(proto_src_files))

  if not proto_src_files and not cc_deps and not import_target:
    raise ValueError(
        f"Proto generation failed: {target.as_label()} no inputs for"
        f" {cc_library_target.as_label()}"
    )

  for dep in plugin_settings.runtime:
    cc_deps.extend(state.get_dep(dep))

  # Construct the output path. This is also the target include dir.
  # ${PROJECT_BINARY_DIR}
  output_dir = get_proto_output_dir(
      _context, proto_info.strip_import_prefix if proto_info else None
  )

  language = (
      plugin_settings.language
      if plugin_settings.language
      else plugin_settings.name
  )
  plugin = ""
  if plugin_settings.plugin:
    cmake_name = state.get_dep(plugin_settings.plugin)
    if len(cmake_name) != 1:
      raise ValueError(
          f"Resolving {plugin_settings.plugin} returned: {cmake_name}"
      )

    cmake_deps.append(cmake_name[0])
    plugin = (
        f"    PLUGIN protoc-gen-{language}=$<TARGET_FILE:{cmake_name[0]}>\n"
    )

  builder = _context.access(CMakeBuilder)
  builder.addtext(f"\n# {cc_library_target.as_label()}")
  emit_cc_library(
      builder,
      cmake_target_pair,
      hdrs=set(),
      srcs=set(proto_src_files),
      deps=set(cc_deps),
  )

  dependencies = ""
  if cmake_deps:
    dependencies = f"    DEPENDENCIES {quote_list(cmake_deps)}\n"

  if proto_src_files:
    # PLUGIN_OPTIONS {quote_list(flags)}
    builder.addtext(f"""
btc_protobuf(
    TARGET {cmake_target_pair.target}
    IMPORT_TARGETS  {import_target}
    LANGUAGE {language}
    GENERATE_EXTENSIONS {quote_list(plugin_settings.exts)}
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR {output_dir}
{plugin}{dependencies})
""")

  _context.add_analyzed_target(
      cc_library_target, TargetInfo(*cmake_target_pair.as_providers())
  )
  return cc_library_target


@register_native_build_rule
def proto_library(
    self: InvocationContext,
    name: str,
    visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  context = self.snapshot()
  target = context.resolve_target(name)
  context.add_rule(
      target,
      lambda: _proto_library_impl(context, target, **kwargs),
      visibility=visibility,
  )


def _proto_library_impl(
    _context: InvocationContext,
    _target: TargetId,
    srcs: Optional[List[RelativeLabel]] = None,
    deps: Optional[List[RelativeLabel]] = None,
    strip_import_prefix: Optional[str] = None,
    **kwargs,
):
  del kwargs
  resolved_srcs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(srcs)
  )
  resolved_deps = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(deps)
  )

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
  import_vars: Optional[str] = None
  import_targets: List[str] = []
  for d in resolved_deps:
    if d in PROTO_REPLACEMENT_TARGETS:
      import_vars = "Protobuf_IMPORT_DIRS"
    state.get_optional_target_info(d)
    import_targets.append(f"{state.generate_cmake_target_pair(d).target}")

  # In order to propagate the includes to our compile targets, each
  # proto_library() becomes a custom CMake target which contains an
  # INTERFACE_INCLUDE_DIRECTORIES property which can be used by the protoc
  # compiler.
  repo = state.workspace.all_repositories.get(_target.repository_id)
  assert repo is not None
  source_dir = repo.source_directory
  bin_dir = repo.cmake_binary_dir
  if strip_import_prefix:
    source_dir = source_dir.joinpath(strip_import_prefix)
    bin_dir = bin_dir.joinpath(strip_import_prefix)

  includes: Set[str] = set()
  for path in state.get_targets_file_paths(resolved_srcs):
    if is_relative_to(pathlib.PurePath(path), source_dir):
      includes.add(source_dir.as_posix())
    if is_relative_to(pathlib.PurePath(path), bin_dir):
      includes.add(bin_dir.as_posix())

  includes: List[str] = list(sorted(includes))
  includes_name = f"{cmake_name}_IMPORT_DIRS"
  includes_literal = "${" + includes_name + "}"
  quoted_srcs = quote_path_list(sorted(set(proto_src_files)), separator=_SEP)

  out = io.StringIO()
  out.write(f"""
# {_target.as_label()}
add_library({cmake_name} INTERFACE)
target_sources({cmake_name} INTERFACE{_SEP}{quoted_srcs})""")
  if import_vars or import_targets:
    out.write(f"""
btc_transitive_import_dirs(
    OUT_VAR {includes_name}
    IMPORT_DIRS {quote_path_list(includes)}
""")
    if import_targets:
      out.write(f"    IMPORT_TARGETS {' '.join(import_targets)}\n")
    if import_vars:
      out.write(f"    IMPORT_VARS {import_vars}\n")
    out.write(")")

  if includes:
    out.write(f"\nlist(APPEND {includes_name} {quote_path_list(includes)})")
  out.write(f"""
set_property(TARGET {cmake_name} PROPERTY INTERFACE_INCLUDE_DIRECTORIES {includes_literal})
""")

  _context.access(CMakeBuilder).addtext(out.getvalue())

  _context.add_analyzed_target(
      _target,
      TargetInfo(
          ProtoLibraryProvider(
              resolved_srcs, resolved_deps, strip_import_prefix
          )
      ),
  )


@register_native_build_rule
def cc_proto_library(
    self: InvocationContext,
    name: str,
    visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  context = self.snapshot()
  target = context.resolve_target(name)
  context.add_rule(
      target,
      lambda: cc_proto_library_impl(context, target, [_CC], **kwargs),
      visibility=visibility,
  )


def cc_proto_library_impl(
    _context: InvocationContext,
    _target: TargetId,
    _plugin_settings: List[PluginSettings],
    deps: Optional[List[RelativeLabel]] = None,
    extra_deps: Optional[List[RelativeLabel]] = None,
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
  library_deps: List[CMakeTarget] = []
  for settings in _plugin_settings:
    for dep_target in resolved_deps:
      lib_target = generate_proto_library_target(
          _context, plugin_settings=settings, target=dep_target
      )
      if lib_target:
        library_deps.extend(state.get_dep(lib_target, alias=False))

  if extra_deps:
    resolved_deps = _context.resolve_target_or_label_list(
        _context.evaluate_configurable_list(extra_deps)
    )
    library_deps.extend(state.get_deps(resolved_deps))

  builder = _context.access(CMakeBuilder)
  builder.addtext(f"\n# {_target.as_label()}")
  emit_cc_library(
      builder,
      cmake_target_pair,
      hdrs=set(),
      srcs=set(),
      deps=set(library_deps),
  )
  _context.add_analyzed_target(
      _target, TargetInfo(*cmake_target_pair.as_providers())
  )
