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
"""CMake implementation of native Bazel build cc_* rules."""

# pylint: disable=relative-beyond-top-level,invalid-name,missing-function-docstring,g-long-lambda

from collections.abc import Collection, Iterable
import enum
import io
import itertools
import pathlib
from typing import Any, NamedTuple, cast

from .cmake_repository import PROJECT_BINARY_DIR
from .cmake_repository import PROJECT_SOURCE_DIR
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetPair
from .emit_alias import emit_alwayslink_alias
from .emit_alias import emit_cc_library_aliases
from .evaluation_state import EvaluationState
from .ordered_set import OrderedSet
from .starlark.bazel_target import PackageId
from .starlark.bazel_target import TargetId
from .starlark.invocation_context import InvocationContext
from .starlark.provider import ProviderTuple
from .util import is_relative_to
from .util import make_relative_path
from .util import partition_by
from .util import PathCollection
from .util import PathLike
from .util import quote_list
from .util import quote_path_list
from .util import quote_unescaped_list
from .variable_substitution import apply_location_substitutions
from .workspace import Workspace

_SEP = "\n        "
_HEADER_SRC_PATTERN = r"\.(?:h|hpp|inc)$"
_ASM_SRC_PATTERN = r"\.(?:s|S|asm)$"
_OBJ_SRC_PATTERN = r"\.(?:o|obj)$"


class CcCommonEnum(enum.Enum):
  INTERFACE = enum.auto()
  STATIC_LIBRARY = enum.auto()
  LIBRARY = enum.auto()
  EXECUTABLE = enum.auto()


class TargetIncludes(NamedTuple):

  system: set[pathlib.PurePath | str]
  public: set[pathlib.PurePath | str]
  private: set[pathlib.PurePath | str]


def default_asm_dialect(workspace: Workspace) -> str:
  """Returns the ASM dialect to use for ASM `srcs` to `cc_*`."""
  if workspace.cmake_vars["CMAKE_CXX_COMPILER_ID"] == "MSVC":
    return "ASM_MASM"
  return "ASM"


def _emit_cc_common_options(
    out: io.StringIO,
    target_name: str,
    asm_dialect: str | None = None,
    link_libraries: Iterable[str] | None = None,
    private_link_libraries: Iterable[str] | None = None,
    copts: Iterable[str] | None = None,
    linkopts: Iterable[str] | None = None,
    defines: Iterable[str] | None = None,
    local_defines: Iterable[str] | None = None,
    includes: Iterable[str] | None = None,
    system_includes: Iterable[str] | None = None,
    private_includes: Iterable[str] | None = None,
    add_dependencies: Iterable[str] | None = None,
    extra_public_compile_options: Iterable[str] | None = None,
    extra_link_options: Iterable[str] | None = None,
    per_file_copts: dict[str, list[str]] | None = None,
    srcs: Iterable[str] | None = None,
    public_srcs: Iterable[str] | None = None,
    emit_enum: CcCommonEnum = CcCommonEnum.LIBRARY,
    cpp_standard: str = "17",
    **kwargs,
):
  """Emits CMake rules for common C++ target options."""
  assert "deps" not in kwargs
  del kwargs

  is_interface = emit_enum == CcCommonEnum.INTERFACE
  public_context = "INTERFACE" if is_interface else "PUBLIC"
  if local_defines is not None and local_defines and not is_interface:
    out.write(
        f"target_compile_definitions({target_name} PRIVATE"
        f" {quote_unescaped_list(local_defines)})\n"
    )
  if defines is not None and defines:
    out.write(
        f"target_compile_definitions({target_name} {public_context}"
        f" {quote_unescaped_list(defines)})\n"
    )
  if copts is not None and copts and not is_interface:
    out.write(
        f"target_compile_options({target_name} PRIVATE {quote_list(copts)})\n"
    )

  if link_libraries or linkopts:
    link_libs: OrderedSet[str] = OrderedSet()
    if link_libraries:
      link_libs.update(link_libraries)

    link_options: OrderedSet[str] = OrderedSet()
    for x in linkopts or []:
      if x.startswith("-l") or x.startswith("-framework"):
        link_libs.add(x)
      else:
        link_options.add(x)
    if link_libs:
      out.write(
          f"target_link_libraries({target_name} {public_context}{_SEP}"
          f"{quote_list(link_libs, separator=_SEP)})\n"
      )
    if link_options:
      out.write(
          f"target_link_options({target_name} {public_context}{_SEP}"
          f"{quote_list(link_options, separator=_SEP)})\n"
      )

  if extra_link_options:
    out.write(
        f"target_link_options({target_name} {public_context}{_SEP}"
        f"{quote_list(extra_link_options, separator=_SEP)})\n"
    )

  if private_link_libraries:
    # MacOS is special; it has a single-pass linker, and CMake PRIVATE link
    # libraries don't propagate to dependents.
    private_link_libs: OrderedSet[str] = OrderedSet(private_link_libraries)
    assert not is_interface, (
        f"{target_name}: interface_only cannot be set with"
        " private_link_libraries"
    )
    out.write(
        f"target_link_libraries({target_name} PRIVATE{_SEP}"
        f"{quote_list(private_link_libs, separator=_SEP)})\n"
    )

  # Only add the include dirs to one of SYSTEM, PUBLIC, or PRIVATE.
  seen_include_dirs = set()

  def _make_include_dirs(includes: Iterable[str]) -> list[str]:
    nonlocal seen_include_dirs
    for x in includes:
      if x not in seen_include_dirs:
        seen_include_dirs.add(x)
        yield f"$<BUILD_INTERFACE:{x}>"

  if system_includes:
    ordered = [x for x in _make_include_dirs(sorted(set(system_includes)))]
    if ordered:
      out.write(
          f"target_include_directories({target_name} SYSTEM {public_context}"
          f"{_SEP}{quote_path_list(ordered, separator=_SEP)})\n"
      )

  if includes:
    ordered = [x for x in _make_include_dirs(sorted(set(includes)))]
    if ordered:
      out.write(
          f"target_include_directories({target_name} {public_context}"
          f"{_SEP}{quote_path_list(ordered, separator=_SEP)})\n"
      )

  if not is_interface and private_includes:
    ordered = [x for x in _make_include_dirs(sorted(set(private_includes)))]
    if ordered:
      out.write(
          f"target_include_directories({target_name} PRIVATE"
          f"{_SEP}{quote_path_list(ordered, separator=_SEP)})\n"
      )

  out.write(
      f"target_compile_features({target_name} {public_context}"
      f" cxx_std_{cpp_standard})\n"
  )
  if add_dependencies:
    out.write(
        f"add_dependencies({target_name} "
        f"{quote_list(sorted(add_dependencies))})\n"
    )
  if extra_public_compile_options:
    out.write(
        f"target_compile_options({target_name} {public_context}{_SEP}"
        f"{quote_list(extra_public_compile_options, separator=_SEP)})\n"
    )
  if per_file_copts:
    for file, copts in per_file_copts.items():
      out.write(
          f"set_source_files_properties({quote_path_list([file])} \n"
          f"  PROPERTIES COMPILE_OPTIONS {quote_list(copts)})\n"
      )

  if srcs or public_srcs:
    non_header_srcs = partition_by(srcs, pattern=_HEADER_SRC_PATTERN)[1]
    asm_srcs = partition_by(
        itertools.chain(non_header_srcs, public_srcs or []),
        pattern=_ASM_SRC_PATTERN,
    )[0]
    if public_srcs:
      out.write(
          f"target_sources({target_name} PUBLIC{_SEP}"
          f"{quote_path_list(public_srcs, separator=_SEP)})\n"
      )
    if non_header_srcs:
      out.write(
          f"target_sources({target_name} PRIVATE{_SEP}"
          f"{quote_path_list(non_header_srcs, separator=_SEP)})\n"
      )
    if asm_srcs:
      if asm_dialect is None:
        raise ValueError(
            f"asm_dialect must be specified for ASM srcs: {asm_srcs!r}"
        )
      out.write(f"""set_source_files_properties(
    {quote_path_list(asm_srcs)}
    PROPERTIES
      LANGUAGE {asm_dialect})\n""")


def construct_cc_includes(
    current_package_id: PackageId,
    *,
    source_directory: pathlib.PurePath,
    cmake_binary_dir: pathlib.PurePath,
    includes: Collection[str] | None = None,
    include_prefix: str | None = None,
    strip_include_prefix: str | None = None,
    hdrs_include_paths: PathCollection | None = None,
    srcs_file_paths: PathCollection | None = None,
) -> TargetIncludes:
  """Returns the set of system and public includes for the configuration.

  By default Bazel generates private includes (-iquote) for the SRCDIR
  and a few other directories.  When `includes` is set, then bazel generates
  public includes (-isystem) which are propagated to callers.

  Here we attempt to generate system includes for CMake based on the
  bazel flags, however it is a best effort technique which, so far, has met
  the needs of cmake translation.

  Args:
    current_package_id: The PackageId of the current Bazel package.
    source_directory: The root of the source tree.
    cmake_binary_dir: The root of the CMake binary output directory.
    includes: A collection of include paths specified in the Bazel rule. These
      are typically treated as system includes.
    include_prefix: An optional prefix to add to the paths of headers.
    strip_include_prefix: An optional prefix to strip from the paths of headers.
    hdrs_include_paths: A collection of file paths for headers (`.h`, `.hpp`,
      etc.) associated with this target, used to determine which include
      directories are actually necessary.
    srcs_file_paths: A collection of file paths for source files (`.cc`, `.cpp`,
      etc.) associated with this target, used to determine private include
      directories.

  Returns:
    A TargetIncludes NamedTuple containing sets of system, public, and private
    include directories.
  """
  if hdrs_include_paths is None:
    hdrs_include_paths = []
  if srcs_file_paths is None:
    srcs_file_paths = []

  system: set[pathlib.PurePath] = set()
  public: set[pathlib.PurePath] = set()
  private: set[pathlib.PurePath] = set()

  current_package_path = pathlib.PurePosixPath(current_package_id.package_name)

  # This include manipulation is a best effort that works for known cases.
  #   https://bazel.build/reference/be/c-cpp#cc_library.includes
  #
  # List of include dirs to be added to the compile line. Subject to
  # "Make variable" substitution. Each string is prepended with the package
  # path and passed to the C++ toolchain for expansion via the "include_paths"
  # CROSSTOOL feature. A toolchain running on a POSIX system with typical
  # feature definitions will produce -isystem path_to_package/include_entry.
  #
  # Unlike COPTS, these flags are added for this rule and every rule that
  # depends on it.
  for include in includes or []:
    # HACK(gRPC): grpc build_system.bzl adds the following includes to
    # all targets; bazel currently requires them, however they interfere in
    # the CMake build, so remove them.
    if (
        current_package_id.repository_id.repository_name == "grpc"
        and include
        in ["src/core/ext/upb-generated", "src/core/ext/upbdefs-generated"]
    ):
      continue

    constructed = str(current_package_path.joinpath(include))
    if constructed[0] == "/":
      constructed = constructed[1:]

    bin_constructed = cmake_binary_dir.joinpath(constructed)
    src_constructed = source_directory.joinpath(constructed)

    system.add(src_constructed)
    system.add(bin_constructed)

  # Track headers not accounted for by the public includes.
  remaining_hdrs: set[PathLike] = set(hdrs_include_paths)

  # For the package foo/bar
  #  - default include path is foo/bar/file.h
  #  - strip_include_prefix=/foo then the include path is bar/file.h
  #  - include_prefix=bar then the include path is bar/file.h
  #
  # bazel_to_cmake only supports composing strip_import_prefix and import_prefix
  # which happens to be a part of the path already, otherwise we'd need to
  # create symbolic links to the files.
  if strip_include_prefix is not None:
    # The prefix to strip from the paths of the headers of this rule.
    #
    # When set, the headers in the hdrs attribute of this rule are accessible
    # at their path with this prefix cut off.
    #
    # If it's a relative path, it's taken as a package-relative one. If it's an
    # absolute one, it's understood as a repository-relative path.
    #
    # The prefix in the include_prefix attribute is added after this prefix is
    # stripped.
    if strip_include_prefix[0] == "/":
      constructed = strip_include_prefix[1:]
    else:
      constructed = str(current_package_path.joinpath(strip_include_prefix))

    src_path = source_directory.joinpath(constructed)
    bin_path = cmake_binary_dir.joinpath(constructed)
    # Only add if existing files are discoverable at the prefix
    for x in hdrs_include_paths:
      c, _ = make_relative_path(x, (src_path, src_path), (bin_path, bin_path))
      if c is not None:
        public.add(c)
        remaining_hdrs.discard(x)

    for x in srcs_file_paths:
      c, _ = make_relative_path(x, (src_path, src_path), (bin_path, bin_path))
      if c is not None:
        private.add(c)

  if include_prefix is not None:
    # The prefix to add to the paths of the headers of this rule.
    #
    # "When set, the headers in the hdrs attribute of this rule are accessible
    # at is the value of this attribute prepended to their repository-relative
    # path."
    #
    # The prefix in the strip_include_prefix attribute is removed before this
    # prefix is added.
    if current_package_id.package_name.endswith(include_prefix):
      # Bazel creates a sandbox (symlink tree) to support the newly composed
      # prefix, but bazel_to_cmake does not, so mimic this behavior by looking
      # at the current package and trying to remove the prefix of the path,
      # thus omitting it from the C search path.
      constructed = current_package_id.package_name[: -len(include_prefix)]

      src_path = source_directory.joinpath(constructed)
      bin_path = cmake_binary_dir.joinpath(constructed)

      # Only add if existing files are discoverable at the prefix
      for x in hdrs_include_paths:
        c, _ = make_relative_path(x, (src_path, src_path), (bin_path, bin_path))
        if c is not None:
          public.add(c)
          remaining_hdrs.discard(x)

      for x in srcs_file_paths:
        c, _ = make_relative_path(x, (src_path, src_path), (bin_path, bin_path))
        if c is not None:
          private.add(c)

  # If there are known include files which are not found under the public
  # include paths, add the source/binary directories to the public includes.
  for x in remaining_hdrs:
    if is_relative_to(pathlib.PurePath(x), cmake_binary_dir):
      if cmake_binary_dir not in system:
        public.add(cmake_binary_dir)
    elif source_directory not in system:
      public.add(source_directory)

  # If there are private includes, also add the source/binary directories.
  for x in itertools.chain(srcs_file_paths, hdrs_include_paths):
    if is_relative_to(pathlib.PurePath(x), cmake_binary_dir):
      private.add(cmake_binary_dir)
    else:
      private.add(source_directory)
  private.difference_update(public)

  return TargetIncludes(system, public, private)


def handle_cc_common_options(
    _context: InvocationContext,
    _src_required=False,
    add_dependencies: Collection[CMakeTarget] | None = None,
    srcs: Collection[TargetId] | None = None,
    deps: Collection[TargetId] | None = None,
    implementation_deps: Collection[TargetId] | None = None,
    includes: Collection[str] | None = None,
    include_prefix: str | None = None,
    strip_include_prefix: str | None = None,
    hdrs_file_paths: Collection[str] | None = None,
    textual_hdrs_file_paths: Collection[str] | None = None,
    _source_directory: pathlib.PurePath | None = None,
    _cmake_binary_dir: pathlib.PurePath | None = None,
    **kwargs,
) -> dict[str, Any]:
  state = _context.access(EvaluationState)
  repo = state.workspace.all_repositories.get(
      _context.caller_package_id.repository_id
  )
  assert repo is not None

  if add_dependencies is None:
    add_dependencies = []
  if _source_directory is None:
    _source_directory = repo.source_directory
  if _cmake_binary_dir is None:
    _cmake_binary_dir = repo.cmake_binary_dir

  srcs_collector = state.collect_targets(srcs)
  srcs_file_paths = list(srcs_collector.file_paths())

  if _src_required and not srcs_file_paths:
    # Sources are required, so add a placeholder source file if there are none.
    srcs_file_paths = [state.get_placeholder_source()]
  elif (
      _src_required
      and not partition_by(srcs_file_paths, pattern=_HEADER_SRC_PATTERN)[1]
  ):
    # Sources are required, and there are sources, but they are all headers.
    srcs_file_paths.append(state.get_placeholder_source())
  elif (
      srcs_file_paths
      and not partition_by(srcs_file_paths, pattern=_OBJ_SRC_PATTERN)[1]
  ):
    # On Windows, CMake does not correctly handle linking libraries containing
    # only nasm (.obj) sources.  Also, when not using the builtin rule,
    # CMake does not handle a library containing only object file as sources.
    # As a workaround, add a placeholder C file.
    srcs_file_paths.append(state.get_placeholder_source())

  copts: list[str] = []

  # Collect per-file copts, which use the target name pf the source file
  # or the generated file name as the key.
  per_file_copts: dict[str, list[str]] = {}
  if state.get_placeholder_source() not in srcs_file_paths:
    all_srcs = set()
    options_to_srcs: dict[str, set[str]] = {}
    for t, srcs in srcs_collector.target_to_paths():
      if t.repository_id == state.workspace.root_repository_id:
        external_dir = ""
      else:
        external_dir = f"external/{t.repository_id.repository_name}"

      srcs = partition_by(srcs, pattern=_HEADER_SRC_PATTERN)[1]
      for src in repo.replace_with_cmake_macro_dirs(srcs):
        all_srcs.add(src)
        my_src = src.replace(PROJECT_SOURCE_DIR, external_dir)
        my_src = my_src.replace(PROJECT_BINARY_DIR, external_dir)
        my_copt = state.workspace.get_per_file_copts(t, my_src)
        if my_copt:
          per_file_copts.setdefault(src, []).extend(my_copt)
          for o in my_copt:
            options_to_srcs.setdefault(o, set()).add(src)

    # Find all per-file which are present for all the source files and promote
    # them to target copts.
    copts = [
        option for option, srcs in options_to_srcs.items() if srcs == all_srcs
    ]

  deps_collector = state.collect_deps(deps)

  link_libraries = OrderedSet(deps_collector.link_libraries())

  # Since Bazel implicitly adds a dependency on the C math library, also add
  # it here.
  link_libraries.add(CMakeTarget("Threads::Threads"))
  if state.workspace.cmake_vars["CMAKE_SYSTEM_NAME"] != "Windows":
    link_libraries.add(CMakeTarget("m"))

  private_link_libraries = OrderedSet()
  if implementation_deps is not None:
    implementation_deps_collector = state.collect_deps(implementation_deps)
    private_link_libraries.update(
        implementation_deps_collector.link_libraries()
    )

  extra_public_compile_options = []

  def add_compile_options(lang: str, options: list[str]):
    for option in options:
      extra_public_compile_options.append(
          f"$<$<COMPILE_LANGUAGE:{lang}>:{option}>"
      )

  copts.extend(state.workspace.copts)

  add_compile_options("C,CXX", copts)
  add_compile_options("CXX", state.workspace.cxxopts)
  add_compile_options("C", state.workspace.conlyopts)

  extra_link_options = []
  for option in state.workspace.linkopts:
    extra_link_options.append(f"$<$<LINK_LANGUAGE:C,CXX>:{option}>")

  # Filter per_file_copts to only include options that are not already covered
  # by copts.
  result = {}
  for k, v in per_file_copts.items():
    seen = set(copts)
    seen_add = seen.add
    my_opt = [x for x in v if not (x in seen or seen_add(x))]
    if my_opt:
      result[k] = my_opt
  per_file_copts = result

  # https://bazel.build/reference/be/make-variables
  # https://bazel.build/reference/be/c-cpp
  # Make variable substitutions apply to the following options:
  # copts, conlyopts, cxxopts, defines
  # includes, linkopts, local_defines
  # For most variables, CMAKE will expand them in the command line, however for
  # $(location) and a few other constructs they need to be handled here.
  def apply_substitutions(k: str, value: Any):
    if not value:
      return value
    new_value = [
        apply_location_substitutions(_context, v, relative_to="") for v in value
    ]
    if new_value != value:
      print(f"Substituted {k}: {value} to {new_value}")
    return new_value

  result: dict[str, Any] = {
      "srcs": repo.replace_with_cmake_macro_dirs(sorted(set(srcs_file_paths))),
      "link_libraries": link_libraries,
      "private_link_libraries": private_link_libraries,
      "add_dependencies": set(
          itertools.chain(
              iter(add_dependencies), srcs_collector.add_dependencies()
          )
      ),
      "extra_public_compile_options": extra_public_compile_options,
      "asm_dialect": default_asm_dialect(state.workspace),
      "extra_link_options": extra_link_options,
      "per_file_copts": per_file_copts,
  }
  for k in ["copts", "linkopts", "defines", "local_defines"]:
    value = kwargs.get(k)
    if value is None:
      result[k] = []
    else:
      resolved = _context.evaluate_configurable_list(cast(Any, value))
      result[k] = apply_substitutions(k, resolved)

  result["defines"].extend(state.workspace.cdefines)

  if include_prefix is not None and strip_include_prefix is not None:
    print(
        f"Warning: package {_context.caller_package_id.package_name} has both"
        f" strip_include_prefix={strip_include_prefix} and"
        f" include_prefix={include_prefix}."
    )

  includes = apply_substitutions("includes", includes)

  target_includes = construct_cc_includes(
      _context.caller_package_id,
      source_directory=_source_directory,
      cmake_binary_dir=_cmake_binary_dir,
      includes=includes,
      include_prefix=include_prefix,
      strip_include_prefix=strip_include_prefix,
      hdrs_include_paths=(hdrs_file_paths or [])
      + (textual_hdrs_file_paths or []),
      srcs_file_paths=srcs_file_paths,
  )
  result["includes"] = repo.replace_with_cmake_macro_dirs(
      target_includes.public
  )
  result["system_includes"] = repo.replace_with_cmake_macro_dirs(
      target_includes.system
  )
  result["private_includes"] = repo.replace_with_cmake_macro_dirs(
      target_includes.private
  )
  result["cpp_standard"] = state.workspace.cpp_standard

  return result


def emit_cc_library(
    out: io.StringIO,
    _cmake_target_pair: CMakeTargetPair,
    *,
    srcs: Collection[str],
    hdrs: Iterable[str],
    alwayslink: bool = False,
    linkstatic: bool = False,
    **kwargs,
) -> ProviderTuple:
  """Generates a C++ library target."""
  header_only = not (partition_by(srcs, pattern=_HEADER_SRC_PATTERN)[1])
  del hdrs

  actual_target = _cmake_target_pair.target
  assert actual_target is not None

  emit_enum = CcCommonEnum.LIBRARY
  if header_only:
    emit_enum = CcCommonEnum.INTERFACE
  elif linkstatic:
    emit_enum = CcCommonEnum.STATIC_LIBRARY

  if alwayslink:
    assert not header_only, (
        f"Target {actual_target} has alwayslink=True but is a header-only"
        " library. alwayslink is only applicable to non-interface libraries."
    )
    actual_target = CMakeTarget(f"{actual_target}.alwayslink")

  # ... what about implementation_deps?
  match (emit_enum):
    case CcCommonEnum.INTERFACE:
      out.write(f"\nadd_library({actual_target} INTERFACE)\n")
    case CcCommonEnum.STATIC_LIBRARY:
      out.write(f"\nadd_library({actual_target} STATIC)\n")
    case _:
      out.write(f"\nadd_library({actual_target})\n")

  if emit_enum != CcCommonEnum.INTERFACE:
    out.write(
        f"""set_property(TARGET {actual_target} PROPERTY LINKER_LANGUAGE "CXX")\n"""
    )

  _emit_cc_common_options(
      out,
      target_name=actual_target,
      emit_enum=emit_enum,
      srcs=sorted(srcs),
      **kwargs,
  )

  # NOTE: We could experiment with the CMake INTERFACE_LINK_LIBRARIES_DIRECT
  # property here, but we might need to query a Bazel provider which records
  # alwayslink information, and that would probably break across projects.
  # Instead, alwayslink cannot be an interface library, (see native_rules_cc.py)

  providers = tuple()
  if alwayslink:
    providers = emit_alwayslink_alias(
        out, _cmake_target_pair.target, actual_target
    )
    actual_target = _cmake_target_pair.target

  emit_cc_library_aliases(out, actual_target, _cmake_target_pair)
  return providers


def emit_cc_binary(
    out: io.StringIO,
    _cmake_target_pair: CMakeTargetPair,
    srcs: set[str],
    **kwargs,
):
  target_name = _cmake_target_pair.target
  assert _cmake_target_pair.alias is not None

  out.write(f'\nadd_executable({target_name} "")\n')
  if _cmake_target_pair.alias != target_name:
    out.write(
        f"add_executable({_cmake_target_pair.alias} ALIAS {target_name})\n"
    )
  _emit_cc_common_options(
      out,
      target_name=target_name,
      srcs=sorted(srcs),
      emit_enum=CcCommonEnum.EXECUTABLE,
      **kwargs,
  )


def emit_cc_test(
    out: io.StringIO,
    _cmake_target_pair: CMakeTargetPair,
    args: list[str] | None = None,
    properties: dict[str, str] | None = None,
    **kwargs,
):
  emit_cc_binary(out, _cmake_target_pair, **kwargs)
  target_name = _cmake_target_pair.target
  args_suffix = ""
  if args:
    args_suffix = " " + " ".join(args)
  out.write(
      f"add_test(NAME {target_name}\n"
      f"      COMMAND {target_name}{args_suffix}\n"
      "      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})\n"
  )
  if properties:
    out.write(f"set_tests_properties({target_name} PROPERTIES\n")
    for k, v in properties.items():
      out.write(f"  {k} {v}\n")
    out.write(")\n")
