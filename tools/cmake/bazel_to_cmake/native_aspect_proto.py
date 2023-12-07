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

"""Aspect-like mechanism to assist in invoking protoc for CMake.

When compiling protos to c++, upb, or other targets, bazel relies on aspects
to apply compilation to a global set of dependencies. Since bazel_to_cmake
doesn't have a global view, this local-view aspects-like mechanism is
configured to generate the required files.  How it works:

All protobuf generators (c++, upb, etc.) are added as ProtoAspectCallables.
The aspect implementation should have the following properties.

* Consistent mapping from proto_library name to output name.
* Gracefully handle blind-references

For each proto_library(), each registered aspect will be invoked. This
aspect is responsible for code generation and perhaps constructing a cc_library,
or other bazel-to-cmake target which can be used later.

Then, specific rules, such as cc_proto_library, can reliably reference the
generated code even from other sub-repositories as long as they are
correctly included in the generated CMake file.
"""

# pylint: disable=invalid-name

import hashlib
import pathlib
from typing import Callable, List, NamedTuple, Optional, Protocol, Tuple

from .cmake_builder import CMakeBuilder
from .cmake_builder import quote_list
from .cmake_target import CMakeLibraryTargetProvider
from .cmake_target import CMakeTarget
from .emit_cc import emit_cc_library
from .evaluation import EvaluationState
from .starlark.bazel_target import RepositoryId
from .starlark.bazel_target import TargetId
from .starlark.common_providers import FilesProvider
from .starlark.common_providers import ProtoLibraryProvider
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.provider import TargetInfo


PROTO_REPO = RepositoryId("com_google_protobuf")
PROTO_COMPILER = PROTO_REPO.parse_target("//:protoc")


class ProtoAspectCallable(Protocol):

  def __call__(
      self,  # ignored
      context: InvocationContext,
      proto_target: TargetId,
      visibility: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    pass


class PluginSettings(NamedTuple):
  name: str
  plugin: Optional[TargetId]
  exts: List[str]
  runtime: List[TargetId]
  language: Optional[str] = None


_PROTO_ASPECT: List[Tuple[str, ProtoAspectCallable]] = []


def add_proto_aspect(name: str, fn: ProtoAspectCallable):
  _PROTO_ASPECT.append((
      name,
      fn,
  ))


def invoke_proto_aspects(
    context: InvocationContext,
    proto_target: TargetId,
    visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  for t in _PROTO_ASPECT:
    t[1](context, proto_target, visibility, **kwargs)


def _get_proto_output_dir(
    _context: InvocationContext,
    out_hash: Optional[str],
    strip_import_prefix: Optional[str],
) -> str:
  """Construct the output path for the proto compiler.

  This is typically a path relative to ${PROJECT_BINARY_DIR} where the
  protocol compiler will output copied protos.
  """
  output_dir = "${PROJECT_BINARY_DIR}"
  if out_hash:
    output_dir = output_dir + f"/{out_hash}"
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
    output_dir = output_dir + f"/{include_path}"
  return output_dir


def btc_protobuf(
    _context: InvocationContext,
    cmake_name: CMakeTarget,
    proto_library_target: TargetId,
    plugin_settings: PluginSettings,
    *,
    cmake_deps: List[CMakeTarget],
    flags: Optional[List[str]] = None,
    out_hash: Optional[str] = None,
) -> str:
  """Generate text to invoke btc_protobuf for a single target."""

  target_info = _context.get_target_info(proto_library_target)
  proto_cmake_target = target_info.get(CMakeLibraryTargetProvider).target

  proto_info = target_info.get(ProtoLibraryProvider)
  assert proto_info is not None

  state = _context.access(EvaluationState)

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

  cmake_deps.extend(state.get_dep(PROTO_COMPILER))
  btc_cmake_deps = list(sorted(set(cmake_deps)))

  # Construct the output path. This is also the target include dir.
  # ${PROJECT_BINARY_DIR}
  output_dir = _get_proto_output_dir(
      _context, out_hash, proto_info.strip_import_prefix
  )

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
    DEPENDENCIES {quote_list(btc_cmake_deps)}
)
"""


def get_aspect_dep(
    _context: InvocationContext,
    t: TargetId,
) -> CMakeTarget:
  # First-party proto references must exist.
  state = _context.access(EvaluationState)
  if _context.caller_package_id.repository_id == t.repository_id:
    target_info = state.get_target_info(t)
  else:
    target_info = state.get_optional_target_info(t)

  if target_info:
    provider = target_info.get(CMakeLibraryTargetProvider)
    if provider:
      return provider.target

  # Get our cmake name; proto libraries need aliases to be referenced
  # from other source trees.
  print(f"Blind reference to {t.as_label()} from {_context.caller_package_id}")
  return state.generate_cmake_target_pair(t).target


def aspect_genproto_library_target(
    _context: InvocationContext,
    *,
    target: TargetId,
    proto_target: TargetId,
    plugin: PluginSettings,
    aspect_dependency: Callable[[TargetId], TargetId],
    extra_deps: Optional[List[TargetId]] = None,
):
  """Emit or return an appropriate TargetId for protos compiled."""
  state = _context.access(EvaluationState)

  # Ensure that the proto_library() target has already been created.
  # First-party rules must exists; this is the common case.
  if (
      target.repository_id == proto_target.repository_id
      or _context.caller_package_id.repository_id == target.repository_id
  ):
    target_info = state.get_target_info(proto_target)
  else:
    target_info = state.get_optional_target_info(proto_target)

  # This library could already have been constructed.
  if state.get_optional_target_info(target) is not None:
    return

  # Get our cmake name; proto libraries need aliases to be referenced
  # from other source trees.
  cmake_target_pair = state.generate_cmake_target_pair(target)

  cc_deps: List[CMakeTarget] = []
  cmake_deps: List[CMakeTarget] = []
  import_target: Optional[CMakeTarget] = None
  proto_src_files: List[str] = []

  # Resolve extra deps first.
  if extra_deps is not None:
    cc_deps.extend(state.get_deps(extra_deps))

  # Run genproto on each dependency.
  proto_info = target_info.get(ProtoLibraryProvider) if target_info else None
  if proto_info is not None:
    for dep in sorted(proto_info.deps):
      cc_deps.append(get_aspect_dep(_context, aspect_dependency(dep)))

    # NOTE: Consider using generator expressions to add to the library target.
    # Something like  $<TARGET_PROPERTY:target,INTERFACE_SOURCES>
    for src in proto_info.srcs:
      proto_src_files.extend(state.get_file_paths(src, cmake_deps))

    import_target = cmake_target_pair.target

  # ... It may be exported as a FilesProvider instead.
  files_info = (
      target_info.get(FilesProvider)
      if target_info is not None and import_target is None
      else None
  )
  if files_info is not None:
    for src in files_info.paths:
      if src.endswith(".proto"):
        proto_src_files.extend(state.get_file_paths(src, cmake_deps))

    if proto_src_files:
      import_target = cmake_target_pair.target

  provider = (
      target_info.get(CMakeLibraryTargetProvider)
      if target_info is not None and import_target is None
      else None
  )
  if provider:
    import_target = provider.target

  if import_target is None:
    # This target is not available; construct an ephemeral reference.
    print(
        f"Blind reference to {target.as_label()} for"
        f" {proto_target.as_label()} from {_context.caller_package_id}"
    )
    return

  proto_src_files = sorted(set(proto_src_files))

  if not proto_src_files and not cc_deps and not import_target:
    raise ValueError(
        f"Proto generation failed: {target.as_label()} no inputs for"
        f" {proto_target.as_label()}"
    )

  out_hash = hashlib.sha1(proto_target.as_label().encode("utf-8")).hexdigest()[
      :8
  ]

  btc_protobuf_txt = ""
  if proto_src_files:
    btc_protobuf_txt = btc_protobuf(
        _context,
        cmake_target_pair.target,
        proto_target,
        plugin,
        cmake_deps=cmake_deps.copy(),
        out_hash=out_hash,
    )

  for dep in plugin.runtime:
    cc_deps.extend(state.get_dep(dep))

  builder = _context.access(CMakeBuilder)
  builder.addtext(f"\n# {target.as_label()}")
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
      target, TargetInfo(*cmake_target_pair.as_providers())
  )
