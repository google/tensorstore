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

import io
import itertools
import pathlib
from typing import Any, Collection, Dict, Iterable, List, Optional, cast

from .cmake_repository import CMakeRepository
from .cmake_repository import PROJECT_BINARY_DIR
from .cmake_repository import PROJECT_SOURCE_DIR
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetPair
from .emit_alias import emit_library_alias
from .evaluation import EvaluationState
from .starlark.bazel_target import PackageId
from .starlark.bazel_target import TargetId
from .starlark.invocation_context import InvocationContext
from .starlark.provider import ProviderTuple
from .starlark.select import Configurable
from .util import is_relative_to
from .util import make_relative_path
from .util import partition_by
from .util import PathCollection
from .util import quote_list
from .util import quote_path_list
from .util import quote_unescaped_list
from .workspace import Workspace

_SEP = "\n        "
_HEADER_SRC_PATTERN = r"\.(?:h|hpp|inc)$"
_ASM_SRC_PATTERN = r"\.(?:s|S|asm)$"


def default_asm_dialect(workspace: Workspace) -> str:
  """Returns the ASM dialect to use for ASM `srcs` to `cc_*`."""
  if workspace.cmake_vars["CMAKE_CXX_COMPILER_ID"] == "MSVC":
    return "ASM_MASM"
  return "ASM"


def _emit_cc_common_options(
    out: io.StringIO,
    target_name: str,
    asm_dialect: Optional[str] = None,
    link_libraries: Optional[Iterable[str]] = None,
    private_link_libraries: Optional[Iterable[str]] = None,
    copts: Optional[Iterable[str]] = None,
    linkopts: Optional[Iterable[str]] = None,
    defines: Optional[Iterable[str]] = None,
    local_defines: Optional[Iterable[str]] = None,
    includes: Optional[Iterable[str]] = None,
    private_includes: Optional[Iterable[str]] = None,
    add_dependencies: Optional[Iterable[str]] = None,
    extra_public_compile_options: Optional[Iterable[str]] = None,
    interface_only: bool = False,
    srcs: Optional[Iterable[str]] = None,
    public_srcs: Optional[Iterable[str]] = None,
    **kwargs,
):
  """Emits CMake rules for common C++ target options."""
  assert "deps" not in kwargs
  del kwargs

  public_context = "INTERFACE" if interface_only else "PUBLIC"
  if local_defines is not None and local_defines and not interface_only:
    out.write(
        f"target_compile_definitions({target_name} PRIVATE"
        f" {quote_unescaped_list(local_defines)})\n"
    )
  if defines is not None and defines:
    out.write(
        f"target_compile_definitions({target_name} {public_context} {quote_unescaped_list(defines)})\n"
    )
  if copts is not None and copts and not interface_only:
    out.write(
        f"target_compile_options({target_name} PRIVATE {quote_list(copts)})\n"
    )

  if link_libraries or linkopts:
    link_libs: List[str] = []
    if link_libraries:
      link_libs.extend(sorted(link_libraries))

    link_options: List[str] = []
    for x in linkopts or []:
      if x.startswith("-l") or x.startswith("-framework"):
        link_libs.append(x)
      else:
        link_options.append(x)
    if link_libs:
      out.write(
          f"target_link_libraries({target_name} {public_context}{_SEP}{quote_list(link_libs, separator=_SEP)})\n"
      )
    if link_options:
      out.write(
          f"target_link_options({target_name} {public_context}{_SEP}{quote_list(link_options, separator=_SEP)})\n"
      )

  if private_link_libraries:
    private_link_libs: List[str] = sorted(private_link_libraries)
    out.write(
        f"target_link_libraries({target_name} PRIVATE{_SEP}{quote_list(private_link_libs, separator=_SEP)})\n"
    )

  include_dirs = [
      f"$<BUILD_INTERFACE:{include_dir}>"
      for include_dir in sorted(set(includes))
  ]
  if include_dirs:
    out.write(
        f"target_include_directories({target_name} {public_context}"
        f"{_SEP}{quote_path_list(include_dirs, separator=_SEP)})\n"
    )
  if not interface_only and private_includes:
    include_dirs = [
        f"$<BUILD_INTERFACE:{include_dir}>"
        for include_dir in sorted(set(private_includes))
    ]
    if include_dirs:
      out.write(
          f"target_include_directories({target_name} PRIVATE"
          f"{_SEP}{quote_path_list(include_dirs, separator=_SEP)})\n"
      )

  out.write(
      f"target_compile_features({target_name} {public_context} cxx_std_17)\n"
  )
  if add_dependencies:
    out.write(
        f"add_dependencies({target_name} "
        f"{quote_list(sorted(add_dependencies))})\n"
    )
  if extra_public_compile_options:
    out.write(
        f"target_compile_options({target_name} {public_context} "
        f"{quote_list(extra_public_compile_options)})\n"
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
          f"{quote_path_list(public_srcs , separator=_SEP)})\n"
      )
    if non_header_srcs:
      out.write(
          f"target_sources({target_name} PRIVATE{_SEP}"
          f"{quote_path_list(non_header_srcs , separator=_SEP)})\n"
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
    includes: Optional[Configurable[List[str]]] = None,
    include_prefix: Optional[str] = None,
    strip_include_prefix: Optional[str] = None,
    known_include_files: Optional[PathCollection] = None,
) -> set[pathlib.PurePath]:
  """Returns the list of system includes for the configuration.

  By default Bazel generates private includes (-iquote) for the SRCDIR
  and a few other directories.  When `includes` is set, then bazel generates
  public includes (-isystem) which are propagated to callers.

  Here we attempt to generate system includes for CMake based on the
  bazel flags, however it is a best effort technique which, so far, has met
  the needs of cmake translation.
  """
  if known_include_files is None:
    known_include_files = []

  include_dirs: set[pathlib.PurePath] = set()
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
        current_package_id.repository_id.repository_name
        == "com_github_grpc_grpc"
        and include
        in ["src/core/ext/upb-generated", "src/core/ext/upbdefs-generated"]
    ):
      continue

    constructed = str(current_package_path.joinpath(include))
    if constructed[0] == "/":
      constructed = constructed[1:]

    include_dirs.add(source_directory.joinpath(constructed))
    include_dirs.add(cmake_binary_dir.joinpath(constructed))

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
    for x in known_include_files:
      (c, _) = make_relative_path(x, (src_path, src_path), (bin_path, bin_path))
      if c is not None:
        include_dirs.add(c)

  if include_prefix is not None:
    # The prefix to add to the paths of the headers of this rule.
    #
    # "When set, the headers in the hdrs attribute of this rule are accessable
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
      for x in known_include_files:
        (c, _) = make_relative_path(
            x, (src_path, src_path), (bin_path, bin_path)
        )
        if c is not None:
          include_dirs.add(c)

  # HACK: Bazel does not add such a fallback, but since three are potential
  # includes CMake needs to include SRC/BIN as interface includes.
  if not include_dirs:
    src_path = source_directory
    bin_path = cmake_binary_dir
    for x in known_include_files:
      (c, _) = make_relative_path(x, (src_path, src_path), (bin_path, bin_path))
      if c is not None:
        include_dirs.add(c)

  return include_dirs


def construct_cc_private_includes(
    repo: CMakeRepository,
    *,
    includes: Optional[List[str]] = None,
    known_include_files: Optional[Iterable[str]] = None,
) -> List[str]:
  if not includes:
    includes = []
  result: List[str] = []
  if PROJECT_SOURCE_DIR not in includes:
    result.append(PROJECT_SOURCE_DIR)
  if PROJECT_BINARY_DIR not in includes:
    for x in known_include_files:
      x_path = pathlib.PurePath(x)
      if is_relative_to(x_path, repo.cmake_binary_dir):
        result.append(PROJECT_BINARY_DIR)
        break
  return result


def handle_cc_common_options(
    _context: InvocationContext,
    src_required=False,
    add_dependencies: Optional[Collection[CMakeTarget]] = None,
    srcs: Optional[Collection[TargetId]] = None,
    deps: Optional[Collection[TargetId]] = None,
    implementation_deps: Optional[Collection[TargetId]] = None,
    includes: Optional[Collection[TargetId]] = None,
    include_prefix: Optional[str] = None,
    strip_include_prefix: Optional[str] = None,
    hdrs_file_paths: Optional[Collection[str]] = None,
    textual_hdrs_file_paths: Optional[Collection[str]] = None,
    _source_directory: Optional[pathlib.PurePath] = None,
    _cmake_binary_dir: Optional[pathlib.PurePath] = None,
    **kwargs,
) -> Dict[str, Any]:
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

  if src_required and not srcs_file_paths:
    srcs_file_paths = [state.get_placeholder_source()]

  deps_collector = state.collect_deps(deps)

  link_libraries = set(deps_collector.link_libraries())

  # Since Bazel implicitly adds a dependency on the C math library, also add
  # it here.
  if state.workspace.cmake_vars["CMAKE_SYSTEM_NAME"] != "Windows":
    link_libraries.add(CMakeTarget("m"))

  link_libraries.add(CMakeTarget("Threads::Threads"))

  if implementation_deps is not None:
    implementation_deps_collector = state.collect_deps(implementation_deps)
    private_link_libraries = set(implementation_deps_collector.link_libraries())
  else:
    private_link_libraries = set()

  extra_public_compile_options = []

  def add_compile_options(lang: str, options: List[str]):
    for option in options:
      extra_public_compile_options.append(
          f"$<$<COMPILE_LANGUAGE:{lang}>:{option}>"
      )

  add_compile_options("C,CXX", state.workspace.copts)
  add_compile_options("CXX", state.workspace.cxxopts)

  result: Dict[str, Any] = {
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
  }
  for k in ["copts", "linkopts", "defines", "local_defines"]:
    value = kwargs.get(k)
    if value is None:
      value = []
    result[k] = _context.evaluate_configurable_list(cast(Any, value))

  result["defines"].extend(state.workspace.cdefines)

  if include_prefix is not None and strip_include_prefix is not None:
    print(
        f"Warning: package {_context.caller_package_id.package_name} has both"
        f" strip_include_prefix={strip_include_prefix} and"
        f" include_prefix={include_prefix}."
    )

  known_include_files = (
      partition_by(srcs_file_paths, pattern=_HEADER_SRC_PATTERN)[0]
      + (hdrs_file_paths or [])
      + (textual_hdrs_file_paths or [])
  )

  result["includes"] = repo.replace_with_cmake_macro_dirs(
      construct_cc_includes(
          _context.caller_package_id,
          source_directory=_source_directory,
          cmake_binary_dir=_cmake_binary_dir,
          includes=includes,
          include_prefix=include_prefix,
          strip_include_prefix=strip_include_prefix,
          known_include_files=known_include_files,
      )
  )

  result["private_includes"] = construct_cc_private_includes(
      repo,
      includes=result["includes"],
      known_include_files=known_include_files,
  )

  return result


def emit_cc_library(
    out: io.StringIO,
    _cmake_target_pair: CMakeTargetPair,
    *,
    srcs: Collection[str],
    hdrs: Iterable[str],
    alwayslink: bool = False,
    header_only: Optional[bool] = None,
    **kwargs,
) -> ProviderTuple:
  """Generates a C++ library target."""
  if header_only is None:
    header_only = not (partition_by(srcs, pattern=_HEADER_SRC_PATTERN)[1])
  del hdrs

  target_name = _cmake_target_pair.target
  assert target_name is not None

  if not header_only:
    out.write(f"""
add_library({target_name})
set_property(TARGET {target_name} PROPERTY LINKER_LANGUAGE "CXX")
""")
  else:
    out.write(f"""
add_library({target_name} INTERFACE)
""")
  _emit_cc_common_options(
      out,
      target_name=target_name,
      interface_only=header_only,
      srcs=sorted(srcs),
      **kwargs,
  )
  if _cmake_target_pair.alias is not None:
    return emit_library_alias(
        out,
        target_name=target_name,
        alias_name=_cmake_target_pair.alias,
        alwayslink=alwayslink,
        interface_only=header_only,
    )
  return tuple()


def emit_cc_binary(
    out: io.StringIO,
    _cmake_target_pair: CMakeTargetPair,
    srcs: set[str],
    **kwargs,
):
  target_name = _cmake_target_pair.target
  assert _cmake_target_pair.alias is not None

  out.write(f"""
add_executable({target_name} "")
add_executable({_cmake_target_pair.alias} ALIAS {target_name})
""")
  _emit_cc_common_options(
      out, target_name=target_name, srcs=sorted(srcs), **kwargs
  )


def emit_cc_test(
    out: io.StringIO,
    _cmake_target_pair: CMakeTargetPair,
    args: Optional[List[str]] = None,
    properties: Optional[Dict[str, str]] = None,
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
