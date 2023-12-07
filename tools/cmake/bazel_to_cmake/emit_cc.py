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

import os
import pathlib
import re
from typing import Any, Dict, Iterable, List, Optional, Set, cast

from .cmake_builder import CMakeBuilder
from .cmake_builder import quote_list
from .cmake_builder import quote_path_list
from .cmake_repository import CMakeRepository
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetPair
from .evaluation import EvaluationState
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.select import Configurable
from .util import is_relative_to
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
    custom_target_deps: Optional[Iterable[str]] = None,
    extra_public_compile_options: Optional[Iterable[str]] = None,
    interface_only: bool = False,
    srcs: Optional[Iterable[str]] = None,
    **kwargs,
):
  """Emits CMake rules for common C++ target options."""
  del kwargs

  # PROJECT_BINARY_DIR and PROJECT_SOURCE_DIR should be in includes
  assert includes is not None
  include_dirs = [
      f"$<BUILD_INTERFACE:{include_dir}>"
      for include_dir in sorted(set(includes))
  ]
  public_context = "INTERFACE" if interface_only else "PUBLIC"
  if local_defines and not interface_only:
    _builder.addtext(
        f"target_compile_definitions({target_name} PRIVATE"
        f" {quote_list(local_defines)})\n"
    )
  if defines:
    _builder.addtext(
        f"target_compile_definitions({target_name} {public_context} {quote_list(defines)})\n"
    )
  if copts and not interface_only:
    _builder.addtext(
        f"target_compile_options({target_name} PRIVATE {quote_list(copts)})\n"
    )

  if deps or linkopts:
    link_libs: List[str] = []
    if deps:
      link_libs.extend(sorted(deps))

    link_options: List[str] = []
    for x in (linkopts or []):
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

  if include_dirs:
    _builder.addtext(
        f"target_include_directories({target_name} {public_context}{_SEP}{quote_path_list(include_dirs, separator=_SEP)})\n"
    )
  _builder.addtext(
      f"target_compile_features({target_name} {public_context} cxx_std_17)\n"
  )
  if custom_target_deps:
    _builder.addtext(
        f"add_dependencies({target_name} {quote_list(sorted(custom_target_deps))})\n"
    )
  if extra_public_compile_options:
    _builder.addtext(
        f"target_compile_options({target_name} {public_context} {quote_list(extra_public_compile_options)})\n"
    )
  if srcs:
    non_header_srcs = [x for x in srcs if not re.search(_HEADER_SRC_PATTERN, x)]
    _builder.addtext(
        f"target_sources({target_name} PRIVATE{_SEP}{quote_path_list(non_header_srcs , separator=_SEP)})\n"
    )

    asm_srcs = [x for x in srcs if re.search(_ASM_SRC_PATTERN, x)]
    if asm_srcs:
      if asm_dialect is None:
        raise ValueError(
            f"asm_dialect must be specified for ASM srcs: {asm_srcs!r}"
        )
      _builder.addtext(f"""set_source_files_properties(
    {quote_path_list(asm_srcs)}
    PROPERTIES
      LANGUAGE {asm_dialect})\n""")


def replace_with_cmake_macro_dirs(
    repo: CMakeRepository, paths: Iterable[str]
) -> List[str]:
  """Substitute reposotory path prefixes with CMake PROJECT_{*}_DIR macros."""
  assert repo is not None

  result: List[str] = []
  for x in paths:
    x_path = pathlib.PurePath(x)
    if is_relative_to(x_path, repo.cmake_binary_dir):
      relative_path = x_path.relative_to(repo.cmake_binary_dir).as_posix()
      if relative_path != ".":
        result.append(f"${{PROJECT_BINARY_DIR}}/{relative_path}")
      else:
        result.append("${PROJECT_BINARY_DIR}")
    elif is_relative_to(x_path, repo.source_directory):
      relative_path = x_path.relative_to(repo.source_directory).as_posix()
      if relative_path != ".":
        result.append(f"${{PROJECT_SOURCE_DIR}}/{relative_path}")
      else:
        result.append("${PROJECT_SOURCE_DIR}")
    else:
      result.append(x)
  return result


def construct_cc_includes(
    _context: InvocationContext,
    *,
    includes: Optional[Configurable[List[str]]] = None,
    include_prefix: Optional[str] = None,
    strip_include_prefix: Optional[str] = None,
    srcs_file_paths: Optional[Iterable[str]] = None,
    hdrs_file_paths: Optional[Iterable[str]] = None,
) -> List[str]:
  state = _context.access(EvaluationState)
  repo = state.workspace.all_repositories.get(
      _context.caller_package_id.repository_id
  )
  assert repo is not None

  # When constructing include dirs, first check the "header-like" files in srcs
  # and make sure that they can be included.
  include_dirs: Set[str] = set()
  add_bare: bool = strip_include_prefix is None and include_prefix is None

  def _try_add(
      paths: Optional[Iterable[str]], repo_paths: List[pathlib.PurePath]
  ):
    if paths is None:
      return
    nonlocal include_dirs
    nonlocal repo
    for x in paths:
      for y in repo_paths:
        if is_relative_to(pathlib.PurePath(x), y):
          include_dirs.add(str(y))

  def _try_add_prefix(paths: Optional[Iterable[str]], prefix: str):
    if not paths:
      return
    if not prefix or prefix == ".":
      nonlocal add_bare
      add_bare = True
      return
    nonlocal repo
    _try_add(
        paths,
        [
            repo.cmake_binary_dir.joinpath(prefix),
            repo.source_directory.joinpath(prefix),
        ],
    )

  # This include manipulation is a best effort that works for known cases.
  #   https://bazel.build/reference/be/c-cpp#cc_library.includes
  #
  current_package_name = _context.caller_package_id.package_name
  relative_package_path = pathlib.PurePosixPath(current_package_name)

  for include in _context.evaluate_configurable_list(includes):
    # HACK(gRPC): grpc build_system.bzl adds the following includes to
    # all targets; bazel currently requires them, however they interfere in
    # the CMake build, so remove them.
    if (
        _context.caller_package_id.repository_id.repository_name
        == "com_github_grpc_grpc"
        and include
        in ["src/core/ext/upb-generated", "src/core/ext/upbdefs-generated"]
    ):
      continue

    include_path = str(
        relative_package_path.joinpath(pathlib.PurePosixPath(include))
    )
    if include_path[0] == "/":
      include_path = include_path[1:]

    _try_add_prefix(hdrs_file_paths, include_path)

  # Assuming a package with files, a/b/c.h:
  # Default include path is "a/b/c.h"
  # When `strip_include_prefix = "/a"` is specified, the include path is "b/c.h"
  # When `include_prefix = "x"` is specified, the include path is "x/a/b/c.h",
  # however if both are specified, then the path would be "x/b/c.h".
  #
  # bazel_to_cmake only supports composing strip_import_prefix and import_prefix
  # which happens to be a part of the path already, otherwise we'd need to
  # create symbolic links to the files.
  if strip_include_prefix is not None:
    # Normalize to a relative prefix.
    if strip_include_prefix[0] == "/":
      _try_add_prefix(hdrs_file_paths, strip_include_prefix[1:])
    else:
      _try_add_prefix(
          hdrs_file_paths,
          str(relative_package_path.joinpath(strip_include_prefix)),
      )

  if include_prefix is not None:
    # "When set, the headers in the hdrs attribute of this rule are accessable
    # at is the value of this attribute prepended to their repository-relative
    # path."
    #
    # Bazel may create a sandbox (symlink tree) to support the newly composed
    # prefix, but bazel_to_cmake does not.
    if current_package_name.endswith(include_prefix):
      computed_prefix = str(
          pathlib.PurePosixPath(current_package_name[: -len(include_prefix)])
      )
      _try_add_prefix(hdrs_file_paths, computed_prefix)

  _try_add(srcs_file_paths, [repo.cmake_binary_dir, repo.source_directory])
  if add_bare:
    _try_add(hdrs_file_paths, [repo.cmake_binary_dir, repo.source_directory])

  return replace_with_cmake_macro_dirs(repo, include_dirs)


def handle_cc_common_options(
    _context: InvocationContext,
    src_required=False,
    custom_target_deps: Optional[List[CMakeTarget]] = None,
    srcs: Optional[Configurable[List[RelativeLabel]]] = None,
    deps: Optional[Configurable[List[RelativeLabel]]] = None,
    includes: Optional[Configurable[List[str]]] = None,
    include_prefix: Optional[str] = None,
    strip_include_prefix: Optional[str] = None,
    hdrs_file_paths: Optional[List[str]] = None,
    textual_hdrs_file_paths: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
  if custom_target_deps is None:
    custom_target_deps = []
  state = _context.access(EvaluationState)

  resolved_srcs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(srcs)
  )
  resolved_deps = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(deps)
  )
  srcs_file_paths = state.get_targets_file_paths(
      resolved_srcs, custom_target_deps
  )

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

  result: Dict[str, Any] = {
      "srcs": set(srcs_file_paths),
      "deps": cmake_deps,
      "custom_target_deps": set(custom_target_deps),
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

  result["includes"] = construct_cc_includes(
      _context,
      includes=includes,
      include_prefix=include_prefix,
      strip_include_prefix=strip_include_prefix,
      srcs_file_paths=[
          x for x in srcs_file_paths if re.search(_HEADER_SRC_PATTERN, x)
      ],
      hdrs_file_paths=set(
          [os.path.dirname(x) for x in (hdrs_file_paths or [])]
          + [os.path.dirname(x) for x in (textual_hdrs_file_paths or [])]
      ),
  )

  return result


def emit_cc_library(
    _builder: CMakeBuilder,
    _cmake_target_pair: CMakeTargetPair,
    srcs: Set[str],
    hdrs: Set[str],
    alwayslink: bool = False,
    header_only: Optional[bool] = None,
    **kwargs,
):
  """Generates a C++ library target."""
  if header_only is None:
    header_only = all(re.search(_HEADER_SRC_PATTERN, x) for x in srcs)
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
    srcs: Set[str],
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
