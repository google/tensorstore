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

import pathlib
from typing import Any, Dict, Iterable, List, Optional, cast

from .cmake_builder import CMakeBuilder
from .cmake_repository import CMakeRepository
from .cmake_repository import PROJECT_BINARY_DIR
from .cmake_repository import PROJECT_SOURCE_DIR
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetPair
from .evaluation import EvaluationState
from .starlark.bazel_target import PackageId
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.select import Configurable
from .util import is_relative_to
from .util import make_relative_path
from .util import partition_by
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
    _builder: CMakeBuilder,
    target_name: str,
    asm_dialect: Optional[str] = None,
    deps: Optional[Iterable[str]] = None,
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
    **kwargs,
):
  """Emits CMake rules for common C++ target options."""
  del kwargs

  public_context = "INTERFACE" if interface_only else "PUBLIC"
  if local_defines is not None and local_defines and not interface_only:
    _builder.addtext(
        f"target_compile_definitions({target_name} PRIVATE"
        f" {quote_unescaped_list(local_defines)})\n"
    )
  if defines is not None and defines:
    _builder.addtext(
        f"target_compile_definitions({target_name} {public_context} {quote_unescaped_list(defines)})\n"
    )
  if copts is not None and copts and not interface_only:
    _builder.addtext(
        f"target_compile_options({target_name} PRIVATE {quote_list(copts)})\n"
    )

  if deps or linkopts:
    link_libs: List[str] = []
    if deps:
      link_libs.extend(sorted(deps))

    link_options: List[str] = []
    for x in linkopts or []:
      if x.startswith("-l") or x.startswith("-framework"):
        link_libs.append(x)
      else:
        link_options.append(x)
    if link_libs:
      _builder.addtext(
          f"target_link_libraries({target_name} {public_context}{_SEP}{quote_list(link_libs, separator=_SEP)})\n"
      )
    if link_options:
      _builder.addtext(
          f"target_link_options({target_name} {public_context}{_SEP}{quote_list(link_options, separator=_SEP)})\n"
      )

  include_dirs = [
      f"$<BUILD_INTERFACE:{include_dir}>"
      for include_dir in sorted(set(includes))
  ]
  if include_dirs:
    _builder.addtext(
        f"target_include_directories({target_name} {public_context}"
        f"{_SEP}{quote_path_list(include_dirs, separator=_SEP)})\n"
    )
  if not interface_only and private_includes:
    include_dirs = [
        f"$<BUILD_INTERFACE:{include_dir}>"
        for include_dir in sorted(set(private_includes))
    ]
    if include_dirs:
      _builder.addtext(
          f"target_include_directories({target_name} PRIVATE"
          f"{_SEP}{quote_path_list(include_dirs, separator=_SEP)})\n"
      )

  _builder.addtext(
      f"target_compile_features({target_name} {public_context} cxx_std_17)\n"
  )
  if add_dependencies:
    _builder.addtext(
        f"add_dependencies({target_name} {quote_list(sorted(add_dependencies))})\n"
    )
  if extra_public_compile_options:
    _builder.addtext(
        f"target_compile_options({target_name} {public_context} {quote_list(extra_public_compile_options)})\n"
    )
  if srcs:
    non_header_srcs = partition_by(*srcs, pattern=_HEADER_SRC_PATTERN)[1]
    _builder.addtext(
        f"target_sources({target_name} PRIVATE{_SEP}{quote_path_list(non_header_srcs , separator=_SEP)})\n"
    )

    asm_srcs = partition_by(*srcs, pattern=_ASM_SRC_PATTERN)[0]
    if asm_srcs:
      if asm_dialect is None:
        raise ValueError(
            f"asm_dialect must be specified for ASM srcs: {asm_srcs!r}"
        )
      _builder.addtext(f"""set_source_files_properties(
    {quote_path_list(asm_srcs)}
    PROPERTIES
      LANGUAGE {asm_dialect})\n""")


def construct_cc_includes(
    repo: CMakeRepository,
    current_package_id: PackageId,
    *,
    includes: Optional[Configurable[List[str]]] = None,
    include_prefix: Optional[str] = None,
    strip_include_prefix: Optional[str] = None,
    known_include_files: Optional[Iterable[str]] = None,
) -> List[str]:
  """Returns the list of system includes for the configuration.

  By default Bazel generates private includes (-iquote) for the SRCDIR
  and a few other directories.  When `includes` is set, then bazel generates
  public includes (-isystem) which are propagated to callers.

  Here we attempt to generate system includes for CMake based on the
  bazel flags, however it is a best effort technique which, so far, has met
  the needs of cmake translation.
  """
  assert repo is not None
  if not known_include_files:
    known_include_files = []

  include_dirs = set()

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

    include_dirs.add(str(repo.source_directory.joinpath(constructed)))
    include_dirs.add(str(repo.cmake_binary_dir.joinpath(constructed)))

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

    src_path = repo.source_directory.joinpath(constructed)
    bin_path = repo.cmake_binary_dir.joinpath(constructed)
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
      src_path = repo.source_directory.joinpath(constructed)
      bin_path = repo.cmake_binary_dir.joinpath(constructed)
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
    src_path = repo.source_directory
    bin_path = repo.cmake_binary_dir
    for x in known_include_files:
      (c, _) = make_relative_path(x, (src_path, src_path), (bin_path, bin_path))
      if c is not None:
        include_dirs.add(c)

  return sorted(repo.replace_with_cmake_macro_dirs(include_dirs))


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
    add_dependencies: Optional[List[CMakeTarget]] = None,
    srcs: Optional[Configurable[List[RelativeLabel]]] = None,
    deps: Optional[Configurable[List[RelativeLabel]]] = None,
    includes: Optional[Configurable[List[str]]] = None,
    include_prefix: Optional[str] = None,
    strip_include_prefix: Optional[str] = None,
    hdrs_file_paths: Optional[List[str]] = None,
    textual_hdrs_file_paths: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
  if add_dependencies is None:
    add_dependencies = []
  state = _context.access(EvaluationState)

  resolved_srcs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(srcs)
  )
  resolved_deps = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(deps)
  )
  srcs_file_paths = [
      str(x) for x in state.get_file_paths(resolved_srcs, add_dependencies)
  ]

  if src_required and not srcs_file_paths:
    srcs_file_paths = [state.get_placeholder_source()]

  cmake_deps = set(state.get_deps(resolved_deps))

  # Since Bazel implicitly adds a dependency on the C math library, also add
  # it here.
  if state.workspace.cmake_vars["CMAKE_SYSTEM_NAME"] != "Windows":
    cmake_deps.add(CMakeTarget("m"))

  cmake_deps.add(CMakeTarget("Threads::Threads"))

  extra_public_compile_options = []

  def add_compile_options(lang: str, options: List[str]):
    for option in options:
      extra_public_compile_options.append(
          f"$<$<COMPILE_LANGUAGE:{lang}>:{option}>"
      )

  add_compile_options("C,CXX", state.workspace.copts)
  add_compile_options("CXX", state.workspace.cxxopts)

  repo = state.workspace.all_repositories.get(
      _context.caller_package_id.repository_id
  )
  assert repo is not None

  result: Dict[str, Any] = {
      "srcs": repo.replace_with_cmake_macro_dirs(sorted(set(srcs_file_paths))),
      "deps": cmake_deps,
      "add_dependencies": set(add_dependencies),
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
      partition_by(*srcs_file_paths, pattern=_HEADER_SRC_PATTERN)[0]
      + (hdrs_file_paths or [])
      + (textual_hdrs_file_paths or [])
  )

  result["includes"] = construct_cc_includes(
      repo,
      _context.caller_package_id,
      includes=_context.evaluate_configurable_list(includes),
      include_prefix=include_prefix,
      strip_include_prefix=strip_include_prefix,
      known_include_files=known_include_files,
  )

  result["private_includes"] = construct_cc_private_includes(
      repo,
      includes=result["includes"],
      known_include_files=known_include_files,
  )

  return result


def emit_cc_library(
    _builder: CMakeBuilder,
    _cmake_target_pair: CMakeTargetPair,
    srcs: set[str],
    hdrs: set[str],
    alwayslink: bool = False,
    header_only: Optional[bool] = None,
    **kwargs,
):
  """Generates a C++ library target."""
  if header_only is None:
    header_only = not (partition_by(*srcs, pattern=_HEADER_SRC_PATTERN)[1])
  del hdrs

  target_name = _cmake_target_pair.target
  assert target_name is not None

  if not header_only:
    _builder.addtext(f"""
add_library({target_name})
set_property(TARGET {target_name} PROPERTY LINKER_LANGUAGE "CXX")
""")
  else:
    _builder.addtext(f"""
add_library({target_name} INTERFACE)
""")
  _emit_cc_common_options(
      _builder,
      target_name=target_name,
      interface_only=header_only,
      srcs=sorted(srcs),
      **kwargs,
  )
  if _cmake_target_pair.alias is not None:
    _builder.add_library_alias(
        target_name=target_name,
        alias_name=_cmake_target_pair.alias,
        alwayslink=alwayslink,
        interface_only=header_only,
    )


def emit_cc_binary(
    _builder: CMakeBuilder,
    _cmake_target_pair: CMakeTargetPair,
    srcs: set[str],
    **kwargs,
):
  target_name = _cmake_target_pair.target
  assert _cmake_target_pair.alias is not None
  _builder.addtext(f"""
add_executable({target_name} "")
add_executable({_cmake_target_pair.alias} ALIAS {target_name})
""")
  _emit_cc_common_options(
      _builder, target_name=target_name, srcs=sorted(srcs), **kwargs
  )


def emit_cc_test(
    _builder: CMakeBuilder,
    _cmake_target_pair: CMakeTargetPair,
    args: Optional[List[str]] = None,
    **kwargs,
):
  emit_cc_binary(_builder, _cmake_target_pair, **kwargs)
  target_name = _cmake_target_pair.target
  args_suffix = ""
  if args:
    args_suffix = " " + " ".join(args)
  _builder.addtext(
      f"""add_test(NAME {target_name} COMMAND {target_name}{args_suffix} WORKING_DIRECTORY ${{CMAKE_CURRENT_SOURCE_DIR}})\n"""
  )
