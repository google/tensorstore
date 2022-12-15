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
import re
from typing import Any, Dict, List, Optional, Set, cast

from .cmake_builder import CMakeBuilder
from .cmake_builder import quote_list
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetPair
from .evaluation import EvaluationState
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.select import Configurable

_SEP = "\n        "
_BASE_INCLUDE_DIRS = ["${PROJECT_SOURCE_DIR}", "${PROJECT_BINARY_DIR}"]


def _emit_cc_common_options(
    _builder: CMakeBuilder,
    target_name: str,
    deps: Optional[Set[str]] = None,
    copts: Optional[List[str]] = None,
    linkopts: Optional[List[str]] = None,
    defines: Optional[List[str]] = None,
    local_defines: Optional[List[str]] = None,
    includes: Optional[List[str]] = None,
    custom_target_deps: Optional[List[str]] = None,
    extra_public_compile_options: Optional[List[str]] = None,
    interface_only: bool = False,
    **kwargs,
):
  """Emits CMake rules for common C++ target options."""
  del kwargs

  # PROJECT_BINARY_DIR and PROJECT_SOURCE_DIR should be in includes
  include_dirs = [
      f"$<BUILD_INTERFACE:{include_dir}>"
      for include_dir in sorted(set(_BASE_INCLUDE_DIRS + (includes or [])))
  ]
  public_context = "INTERFACE" if interface_only else "PUBLIC"
  if local_defines and not interface_only:
    _builder.addtext(
        f"target_compile_definitions({target_name} PRIVATE {quote_list(local_defines)})\n"
    )
  if defines:
    _builder.addtext(
        f"target_compile_definitions({target_name} {public_context} {quote_list(defines)})\n"
    )
  if copts and not interface_only:
    _builder.addtext(
        f"target_compile_options({target_name} PRIVATE {quote_list(copts)})\n")
  if deps or linkopts:
    link_libs: List[str] = []
    if deps:
      link_libs.extend(sorted(deps))
    if linkopts:
      link_libs.extend(linkopts)
    _builder.addtext(
        f"target_link_libraries({target_name} {public_context}{_SEP}{quote_list(link_libs, separator=_SEP)})\n"
    )
  _builder.addtext(
      f"target_include_directories({target_name} {public_context}{_SEP}{quote_list(include_dirs, separator=_SEP)})\n"
  )
  _builder.addtext(
      f"target_compile_features({target_name} {public_context} cxx_std_17)\n")
  if custom_target_deps:
    _builder.addtext(
        f"add_dependencies({target_name} {quote_list(custom_target_deps)})\n")
  if extra_public_compile_options:
    _builder.addtext(
        f"target_compile_options({target_name} {public_context} {quote_list(extra_public_compile_options)})\n"
    )


def handle_cc_common_options(
    _context: InvocationContext,
    src_required=False,
    custom_target_deps: Optional[List[CMakeTarget]] = None,
    srcs: Optional[Configurable[List[RelativeLabel]]] = None,
    deps: Optional[Configurable[List[RelativeLabel]]] = None,
    includes: Optional[Configurable[List[str]]] = None,
    include_prefix: Optional[str] = None,
    strip_include_prefix: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
  if custom_target_deps is None:
    custom_target_deps = []
  state = _context.access(EvaluationState)

  resolved_srcs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(srcs))
  resolved_deps = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(deps))
  srcs_file_paths = state.get_targets_file_paths(resolved_srcs,
                                                 custom_target_deps)

  if src_required and not srcs_file_paths:
    srcs_file_paths = [state.get_dummy_source()]

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
          f"$<$<COMPILE_LANGUAGE:{lang}>:{option}>")

  add_compile_options("C,CXX", state.workspace.copts)
  add_compile_options("CXX", state.workspace.cxxopts)

  result: Dict[str, Any] = {
      "srcs": set(srcs_file_paths),
      "deps": cmake_deps,
      "custom_target_deps": set(custom_target_deps),
      "extra_public_compile_options": extra_public_compile_options,
  }
  for k in ["copts", "linkopts", "defines", "local_defines"]:
    value = kwargs.get(k)
    if value is None:
      value = []
    result[k] = _context.evaluate_configurable_list(cast(Any, value))

  result["defines"].extend(state.workspace.cdefines)

  # This include manipulation is a best effort that works for known cases.
  #   https://bazel.build/reference/be/c-cpp#cc_library.includes
  #
  # When absolute, includes and strip_include_prefix are repository relative,
  # otherwise they are package relative.
  current_package_name = _context.caller_package_id.package_name

  include_dirs: List[str] = []

  relative_package_path = pathlib.PurePosixPath(current_package_name)
  for include in _context.evaluate_configurable_list(includes):
    # gRPC hack: grpc build_system.bzl adds the following includes to
    # all targets; bazel currently requires them, however they interfere in
    # the CMake build, so remove them.
    if (_context.caller_package_id.repository_id.repository_name
        == "com_github_grpc_grpc" and include
        in ["src/core/ext/upb-generated", "src/core/ext/upbdefs-generated"]):
      continue
    include_path = str(
        relative_package_path.joinpath(pathlib.PurePosixPath(include)))
    if include_path[0] == "/":
      include_path = include_path[1:]
    include_dirs.extend(f"{i}/{include_path}" for i in _BASE_INCLUDE_DIRS)

  if strip_include_prefix is not None:
    include_path = str(
        relative_package_path.joinpath(
            pathlib.PurePosixPath(strip_include_prefix)))
    if include_path[0] == "/":
      include_path = include_path[1:]
    include_dirs.extend(f"{i}/{include_path}" for i in _BASE_INCLUDE_DIRS)

  if include_prefix is not None:
    # "When set, the headers in the hdrs attribute of this rule are accessable
    # at is the value of this attribute prepended to their repository-relative
    # path."
    #
    # Bazel may create a symlink tree to support this and strip_include_prefix;
    # for known cases this appears to work.
    if current_package_name.endswith(include_prefix):
      computed_prefix = str(
          pathlib.PurePosixPath(current_package_name[:-len(include_prefix)]))
      include_dirs.extend(f"{i}/{computed_prefix}" for i in _BASE_INCLUDE_DIRS)

  result["includes"] = include_dirs
  return result


def emit_cc_library(
    _builder: CMakeBuilder,
    _cmake_target_pair: CMakeTargetPair,
    srcs: Set[str],
    hdrs: Set[str],
    alwayslink: bool = False,
    **kwargs,
):
  """Generates a C++ library target."""
  cc_srcs = sorted([x for x in srcs if not re.search(r"\.(?:h|inc)$", x)])
  header_only = not cc_srcs
  del hdrs

  target_name = _cmake_target_pair.target
  assert target_name is not None

  if not header_only:
    _builder.addtext(f"""
add_library({target_name})
target_sources({target_name} PRIVATE{_SEP}{quote_list(cc_srcs , separator=_SEP)})
set_property(TARGET {target_name} PROPERTY LINKER_LANGUAGE "CXX")
""")
  else:
    _builder.addtext(f"""
add_library({target_name} INTERFACE)
""")
  _emit_cc_common_options(
      _builder, target_name=target_name, interface_only=header_only, **kwargs)
  if _cmake_target_pair.alias is not None:
    _builder.add_library_alias(
        target_name=target_name,
        alias_name=_cmake_target_pair.alias,
        alwayslink=alwayslink,
        interface_only=header_only,
    )


def emit_cc_binary(_builder: CMakeBuilder, _cmake_target_pair: CMakeTargetPair,
                   srcs: Set[str], **kwargs):
  target_name = _cmake_target_pair.target
  assert _cmake_target_pair.alias is not None
  _builder.addtext(f"""
add_executable({target_name} "")
add_executable({_cmake_target_pair.alias} ALIAS {target_name})
target_sources({target_name} PRIVATE{_SEP}{quote_list(sorted(srcs), separator=_SEP)})
""")
  _emit_cc_common_options(_builder, target_name=target_name, **kwargs)


def emit_cc_test(_builder: CMakeBuilder,
                 _cmake_target_pair: CMakeTargetPair,
                 args: Optional[List[str]] = None,
                 **kwargs):
  emit_cc_binary(_builder, _cmake_target_pair, **kwargs)
  target_name = _cmake_target_pair.target
  args_suffix = ""
  if args:
    args_suffix = " " + " ".join(args)
  _builder.addtext(
      f"""add_test(NAME {target_name} COMMAND {target_name}{args_suffix} WORKING_DIRECTORY ${{CMAKE_CURRENT_SOURCE_DIR}})\n"""
  )
