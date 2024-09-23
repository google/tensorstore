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
import io
import pathlib
from typing import Callable, Iterable, List, NamedTuple, Optional, Set, Tuple

from .cmake_builder import CMakeBuilder
from .cmake_provider import CMakeAddDependenciesProvider
from .cmake_provider import CMakePackageDepsProvider
from .cmake_provider import default_providers
from .cmake_provider import make_providers
from .cmake_repository import PROJECT_BINARY_DIR
from .cmake_target import CMakeTarget
from .emit_cc import emit_cc_library
from .emit_cc import handle_cc_common_options
from .evaluation import EvaluationState
from .provider_util import ProviderCollection
from .starlark.bazel_target import RepositoryId
from .starlark.bazel_target import TargetId
from .starlark.common_providers import FilesProvider
from .starlark.common_providers import ProtoLibraryProvider
from .starlark.invocation_context import InvocationContext
from .starlark.provider import TargetInfo
from .util import exactly_one
from .util import partition_by
from .util import PathIterable
from .util import PathLike
from .util import quote_list
from .util import quote_path
from .util import quote_path_list
from .util import quote_string


PROTO_REPO = RepositoryId("com_google_protobuf")
PROTO_COMPILER = PROTO_REPO.parse_target("//:protoc")
_HEADER_SRC_PATTERN = r"\.(?:h|hpp|inc)$"


class PluginSettings(NamedTuple):
  name: str
  plugin: Optional[TargetId]
  exts: List[str]
  runtime: List[TargetId]
  aspectdeps: Callable[[TargetId], List[TargetId]]
  language: Optional[str] = None


##############################################################################


def btc_protobuf(
    _context: InvocationContext,
    out: io.StringIO,
    *,
    plugin_settings: PluginSettings,
    proto_src: PathLike,
    generated_files: PathIterable,
    cmake_depends: Optional[Iterable[CMakeTarget]] = None,
    flags: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    import_targets: Optional[Iterable[CMakeTarget]] = None,
):
  """Generate text to invoke btc_protobuf for a single target."""
  if not output_dir:
    output_dir = PROJECT_BINARY_DIR

  if not cmake_depends:
    cmake_depends = set()
  else:
    cmake_depends = set(cmake_depends)
  cmake_depends.add("protobuf::protoc")
  cmake_depends.add(proto_src)

  state = _context.access(EvaluationState)
  collector = ProviderCollection()

  plugin = ""
  if plugin_settings.plugin:
    plugin_name = exactly_one(
        state.collect_deps([plugin_settings.plugin], collector).targets()
    )
    cmake_depends.add(plugin_name)
    plugin = (
        "\n   "
        f" --plugin=protoc-gen-{plugin_settings.language}=$<TARGET_FILE:{plugin_name}>"
    )

  state.collect_deps([PROTO_COMPILER], collector)
  cmake_depends.update(collector.targets())

  # Build the flags for the plugin.
  if flags:
    flags = ",".join(flags)
    flags = f"--{plugin_settings.language}_out={flags}:{output_dir}"
  else:
    flags = f"--{plugin_settings.language}_out={output_dir}"

  _sep = "\n    "

  mkdirs = set()

  def _generated_files():
    for x in sorted(generated_files):
      mkdirs.add(pathlib.PurePath(x).parent)
      yield x

  quoted_paths = quote_path_list(_generated_files(), separator=_sep)

  # Emit.
  if mkdirs:
    out.write(f"file(MAKE_DIRECTORY {quote_path_list(sorted(mkdirs))})\n")
  out.write(f"""add_custom_command(
OUTPUT
    {quoted_paths}
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional{plugin}""")
  for x in sorted(set(import_targets)):
    _prop = f"$<TARGET_PROPERTY:{x},INTERFACE_INCLUDE_DIRECTORIES>"
    _expr = f"-I$<JOIN:{_prop},$<SEMICOLON>-I>"
    # _ifexists = f"$<$<TARGET_EXISTS:{x}>:$<$<BOOL:${_prop}>:{_expr}>>"
    out.write(f"\n    {quote_string(_expr)}")
  out.write(f"""
    {quote_string(flags)}
    {quote_path(proto_src)}
DEPENDS
    {quote_list(sorted(cmake_depends), separator=_sep)}
COMMENT "Running protoc {plugin_settings.name} on {proto_src}"
COMMAND_EXPAND_LISTS
VERBATIM
)\n""")


##############################################################################
# Single Protoaspect details
##############################################################################
#
# Bazel proto_libraries() can reference soruce files multiple times, so
# the aspect actually needs to generate a target per source file.
#
def _assert_is_proto(src: TargetId):
  assert src.target_name.endswith(
      ".proto"
  ), f"{src.as_label()} must end in .proto"


def maybe_augment_output_dir(
    _context: InvocationContext,
    proto_library_provider: ProtoLibraryProvider,
    output_dir: pathlib.PurePath,
):
  """If necessary, augment the output_dir with the strip_import_prefix."""
  if proto_library_provider.strip_import_prefix:
    include_path = str(
        pathlib.PurePosixPath(_context.caller_package_id.package_name).joinpath(
            proto_library_provider.strip_import_prefix
        )
    )
    if include_path[0] == "/":
      include_path = include_path[1:]
    return output_dir.joinpath(include_path)
  return output_dir


def singleproto_aspect_target(
    src: TargetId,
    plugin_settings: PluginSettings,
) -> TargetId:
  _assert_is_proto(src)
  hash_id = hashlib.sha1(src.as_label().encode("utf-8")).hexdigest()[:8]
  return src.get_target_id(f"aspect_{plugin_settings.name}__{hash_id}")


def plugin_generated_files(
    src: TargetId,
    plugin_settings: PluginSettings,
    binary_dir: pathlib.PurePath,
) -> List[Tuple[TargetId, pathlib.PurePath]]:
  """Returns a list of (target, path)... generated by plugin_settings."""
  _assert_is_proto(src)
  target_name = src.target_name.removesuffix(".proto")

  def _mktuple(ext: str) -> Tuple[TargetId, pathlib.PurePath]:
    generated_target = src.get_target_id(f"{target_name}{ext}")
    generated_path = generated_target.as_rooted_path(binary_dir)
    return (generated_target, generated_path)

  return [_mktuple(ext) for ext in plugin_settings.exts]


def aspect_genproto_singleproto(
    _context: InvocationContext,
    *,
    aspect_target: TargetId,
    src: TargetId,
    proto_library_provider: ProtoLibraryProvider,
    plugin_settings: PluginSettings,
    output_dir: pathlib.PurePath,
):
  _assert_is_proto(src)
  state = _context.access(EvaluationState)
  repo = state.workspace.all_repositories.get(
      _context.caller_package_id.repository_id
  )
  assert repo is not None

  # This library could already have been constructed.
  if state.get_optional_target_info(aspect_target) is not None:
    return

  # Get our cmake name; proto libraries need aliases to be referenced
  # from other source trees.
  cmake_target_pair = state.generate_cmake_target_pair(
      aspect_target, alias=False
  )

  src_collector = state.collect_targets([src])
  proto_src_files: List[str] = list(set(src_collector.file_paths()))
  if not proto_src_files:
    raise ValueError(
        f"Proto generation failed: {src.as_label()} has no proto srcs"
    )
  assert len(proto_src_files) == 1

  proto_cmake_target = state.generate_cmake_target_pair(
      proto_library_provider.bazel_target
  ).target
  import_targets = set(
      state.collect_deps(proto_library_provider.deps).link_libraries()
  )
  import_targets.add(proto_cmake_target)

  # Emit the generated files.
  generated_files = []
  for t in plugin_generated_files(
      src,
      plugin_settings,
      output_dir,
  ):
    _context.add_analyzed_target(
        t[0],
        TargetInfo(
            *make_providers(
                cmake_target_pair,
                CMakePackageDepsProvider,
                CMakeAddDependenciesProvider,
            ),
            FilesProvider([t[1]]),
        ),
    )
    generated_files.append(t[1])

  output_dir = maybe_augment_output_dir(
      _context, proto_library_provider, output_dir
  )

  # Emit.
  out = io.StringIO()
  out.write(
      f"\n# {aspect_target.as_label()}\n"
      f"# genproto {plugin_settings.name} {src.as_label()}\n"
  )
  btc_protobuf(
      _context,
      out,
      plugin_settings=plugin_settings,
      # varname=str(cmake_target_pair.target),
      proto_src=proto_src_files[0],
      generated_files=generated_files,
      import_targets=import_targets,
      cmake_depends=sorted(src_collector.add_dependencies()),
      output_dir=repo.replace_with_cmake_macro_dirs([output_dir])[0],
  )

  sep = "\n    "
  quoted_outputs = quote_path_list(
      repo.replace_with_cmake_macro_dirs(generated_files), sep
  )
  out.write(
      f"add_custom_target({cmake_target_pair.target} DEPENDS{sep}{quoted_outputs})\n"
  )
  _context.access(CMakeBuilder).addtext(out.getvalue())

  _context.add_analyzed_target(
      aspect_target,
      TargetInfo(
          *make_providers(
              cmake_target_pair,
              CMakeAddDependenciesProvider,
              CMakePackageDepsProvider,
          ),
          FilesProvider(generated_files),
      ),
  )


##############################################################################


def aspect_genproto_library_target(
    _context: InvocationContext,
    *,
    target: TargetId,
    proto_target: TargetId,
    plugin_settings: PluginSettings,
):
  """Emit or return an appropriate TargetId for protos compiled."""
  state = _context.access(EvaluationState)
  repo = state.workspace.all_repositories.get(
      _context.caller_package_id.repository_id
  )
  assert repo is not None

  # Ensure that the proto_library() target has already been created.
  # First-party rules must exists; this is the common case.
  if (
      target.repository_id == proto_target.repository_id
      or _context.caller_package_id.repository_id == target.repository_id
  ):
    proto_target_info = state.get_target_info(proto_target)
  else:
    proto_target_info = state.get_optional_target_info(proto_target)

  # This library could already have been constructed.
  if state.get_optional_target_info(target) is not None:
    return

  if not proto_target_info or (
      proto_target_info.get(ProtoLibraryProvider) is None
      and proto_target_info.get(CMakeAddDependenciesProvider) is None
      and proto_target_info.get(FilesProvider) is None
  ):
    # This target is not available; construct an ephemeral reference.
    print(
        f"Blind reference to {target.as_label()} for"
        f" {proto_target.as_label()} from {_context.caller_package_id}"
    )
    return

  proto_library_provider = proto_target_info.get(ProtoLibraryProvider)
  assert proto_library_provider is not None

  # Resovle aspect deps, excluding self.
  aspect_deps: Set[TargetId] = set()
  aspect_deps.update(plugin_settings.aspectdeps(proto_target))
  for d in proto_library_provider.deps:
    aspect_deps.update(plugin_settings.aspectdeps(d))
  aspect_deps.update(plugin_settings.runtime)
  if target in aspect_deps:
    aspect_deps.remove(target)

  cmake_target_pair = state.generate_cmake_target_pair(target)

  # Build per-file stuff.
  output_dir = repo.cmake_binary_dir.joinpath(f"_gen_{plugin_settings.name}")

  aspect_srcs: set[TargetId] = set()
  for src in proto_library_provider.srcs:
    aspect_target = singleproto_aspect_target(src, plugin_settings)
    aspect_srcs.add(aspect_target)
    aspect_genproto_singleproto(
        _context,
        aspect_target=aspect_target,
        src=src,
        proto_library_provider=proto_library_provider,
        plugin_settings=plugin_settings,
        output_dir=output_dir,
    )

  aspect_dir = repo.replace_with_cmake_macro_dirs(
      [maybe_augment_output_dir(_context, proto_library_provider, output_dir)]
  )[0]

  # Now emit a cc_library for each proto_library
  srcs_collector = state.collect_targets(aspect_srcs)
  split_srcs = partition_by(
      srcs_collector.file_paths(), pattern=_HEADER_SRC_PATTERN
  )

  common_options = handle_cc_common_options(
      _context,
      src_required=True,
      add_dependencies=set(srcs_collector.add_dependencies()),
      srcs=aspect_srcs,
      deps=aspect_deps,
      includes=None,
      include_prefix=proto_library_provider.import_prefix,
      strip_include_prefix=proto_library_provider.strip_import_prefix,
      hdrs_file_paths=split_srcs[0],
      _source_directory=repo.source_directory,
      _cmake_binary_dir=output_dir,
  )
  del common_options["private_includes"]

  if aspect_dir not in common_options["includes"]:
    common_options["includes"].append(aspect_dir)

  out = io.StringIO()
  out.write(
      f"\n# {target.as_label()}"
      f"\n# aspect {plugin_settings.name} {proto_target.as_label()}"
  )
  emit_cc_library(
      out,
      cmake_target_pair,
      hdrs=split_srcs[0],
      public_srcs=repo.replace_with_cmake_macro_dirs(sorted(split_srcs[0])),
      **common_options,
  )
  _context.access(CMakeBuilder).addtext(out.getvalue())
  _context.add_analyzed_target(
      target, TargetInfo(*default_providers(cmake_target_pair))
  )
