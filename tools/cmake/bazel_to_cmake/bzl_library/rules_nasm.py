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
"""CMake implementation of "@com_google_tensorstore//bazel:rules_nasm.bzl"."""

# pylint: disable=relative-beyond-top-level,invalid-name,missing-function-docstring,missing-class-docstring

import hashlib
import os
import pathlib
from typing import List, Optional, Set, cast

from ..cmake_builder import CMakeBuilder
from ..cmake_builder import quote_list
from ..cmake_builder import quote_path
from ..cmake_builder import quote_path_list
from ..cmake_builder import quote_string
from ..cmake_target import CMakeTarget
from ..evaluation import EvaluationState
from ..starlark.bazel_globals import BazelGlobals
from ..starlark.bazel_globals import register_bzl_library
from ..starlark.bazel_target import TargetId
from ..starlark.invocation_context import InvocationContext
from ..starlark.invocation_context import RelativeLabel
from ..starlark.provider import TargetInfo
from ..starlark.select import Configurable
from ..util import cmake_is_true


@register_bzl_library(
    "@com_google_tensorstore//bazel:rules_nasm.bzl", build=True)
class RulesNasmLibrary(BazelGlobals):

  def bazel_nasm_library(self,
                         name: str,
                         visibility: Optional[List[RelativeLabel]] = None,
                         **kwargs):
    context = self._context.snapshot()
    target = context.resolve_target(name)
    context.add_rule(
        target,
        lambda: _nasm_library_impl(context, target, **kwargs),
        visibility=visibility)


def _nasm_library_impl(
    _context: InvocationContext,
    _target: TargetId,
    srcs: Optional[Configurable[List[RelativeLabel]]] = None,
    flags: Optional[Configurable[List[str]]] = None,
    includes: Optional[Configurable[List[RelativeLabel]]] = None,
    alwayslink: bool = False,
    **kwargs,
):
  del kwargs
  state = _context.access(EvaluationState)

  # Construct a package prefix string to compare against the include target.
  package_prefix_str = _context.caller_repository_name
  if _context.caller_package_id.package_name:
    package_prefix_str += "/"

  cmake_target_pair = state.generate_cmake_target_pair(_target)
  cmake_deps: List[CMakeTarget] = []

  resolved_flags = _context.evaluate_configurable(flags or [])

  resolved_srcs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(srcs or []))
  src_files = set(state.get_targets_file_paths(resolved_srcs, cmake_deps))

  resolved_includes = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(includes or []))

  all_includes = set()
  for include_target in resolved_includes:
    # Get include target relative to current package, so that if `includes =
    # ["a/b/whatever.asm"]` then it can be found as either "whatever.asm" or
    # "a/b/whatever.asm".
    include_target_str = include_target.as_label()
    if include_target_str.startswith(package_prefix_str):
      relative_target = include_target_str[len(package_prefix_str):].replace(
          ":", "/")

    for include_file in state.get_file_paths(include_target, cmake_deps):
      all_includes.add(str(pathlib.PurePosixPath(include_file).parent))
      if relative_target and include_file.endswith(relative_target):
        all_includes.add(include_file[:-len(relative_target)].rstrip("/"))

  use_builtin_rule = True

  workspace = state.workspace
  if (cmake_is_true(workspace.cmake_vars.get("MSVC_IDE")) or
      cmake_is_true(workspace.cmake_vars.get("XCODE"))):
    # The "Xcode" generator does not support nasm at all.  The "Visual Studio *"
    # generators support nasm but pass "/DWIN32" which is problematic for some
    # packages.
    #
    # TODO(jbms): Due to https://gitlab.kitware.com/cmake/cmake/-/issues/24079,
    # this rule does not work correctly with the Xcode generator if
    # `alwayslink=True`.
    use_builtin_rule = False
  # On Windows, CMake does not correctly handle linking of libraries containing
  # only nasm sources.  Also, when not using the builtin rule, CMake does not
  # handle a library containing only object file as sources.  As a workaround,
  # add a dummy C file.
  dummy_source: Optional[str] = None
  if workspace.cmake_vars[
      "CMAKE_CXX_COMPILER_ID"] == "MSVC" or not use_builtin_rule:
    dummy_source = state.get_dummy_source()
  _emit_nasm_library(
      _context.access(CMakeBuilder),
      target_name=cmake_target_pair.target,
      alias_name=cmake_target_pair.alias,
      cmake_deps=cmake_deps,
      srcs=src_files,
      flags=resolved_flags,
      includes=all_includes,
      alwayslink=alwayslink,
      dummy_source=dummy_source,
      use_builtin_rule=use_builtin_rule,
  )
  _context.add_analyzed_target(_target,
                               TargetInfo(*cmake_target_pair.as_providers()))


_EMIT_YASM_CHECK = """
get_filename_component(_nasm_compiler_barename "${CMAKE_ASM_NASM_COMPILER}" NAME)
if (_nasm_compiler_barename STREQUAL "yasm")
  message(WARNING "CMake found YASM assembler. Please install 'nasm' instead.")
endif()
unset(_nasm_compiler_barename)

"""


def _emit_nasm_library(
    _builder: CMakeBuilder,
    target_name: CMakeTarget,
    alias_name: Optional[CMakeTarget],
    cmake_deps: List[CMakeTarget],
    srcs: Set[str],
    includes: Set[str],
    flags: List[str],
    dummy_source: Optional[str],
    alwayslink: bool,
    use_builtin_rule: bool,
):
  """Generates an NASM library target."""
  all_srcs = sorted(srcs)
  dummy_sources = [dummy_source] if dummy_source is not None else []
  _builder.addtext(f"add_library({target_name})\n")
  if use_builtin_rule:
    _builder.addtext(_EMIT_YASM_CHECK, unique=True)
    _builder.addtext(
        f"""target_sources({target_name} PRIVATE {quote_list(all_srcs + dummy_sources)})
target_include_directories({target_name} PRIVATE {quote_list(sorted(includes))})
set_source_files_properties(
    {quote_list(all_srcs)}
    PROPERTIES
      LANGUAGE ASM_NASM
      COMPILE_OPTIONS {quote_string(";".join(flags))})\n""")
    if cmake_deps:
      _builder.addtext(
          f"add_dependencies({target_name} {quote_list(sorted(cmake_deps))})\n")
  else:
    all_flags = list(flags)
    for include in includes:
      all_flags.append("-I" + include)
    all_src_obj_exprs = []
    for src in sorted(srcs):
      src_obj_base = os.path.basename(src) + "_" + hashlib.sha256(
          src.encode("utf-8")).hexdigest()[:16]
      src_obj_expr = f"${{CMAKE_CURRENT_BINARY_DIR}}/{src_obj_base}${{CMAKE_C_OUTPUT_EXTENSION}}"
      all_src_obj_exprs.append(src_obj_expr)
      _builder.addtext(f"""
add_custom_command(
  OUTPUT {quote_string(src_obj_expr)}
  DEPENDS {quote_path_list(all_srcs + cast(List[str], cmake_deps))}
  COMMAND ${{CMAKE_ASM_NASM_COMPILER}}
          -f ${{CMAKE_ASM_NASM_OBJECT_FORMAT}}
          ${{CMAKE_ASM_NASM_FLAGS}}
          {quote_list(all_flags)}
          {quote_path(src)}
          -o {quote_string(src_obj_expr)}
)
""")

    _builder.addtext(
        f"""target_sources({target_name} PRIVATE {quote_list(all_src_obj_exprs + dummy_sources)})\n"""
    )
  if alias_name:
    _builder.add_library_alias(
        target_name=target_name,
        alias_name=alias_name,
        alwayslink=alwayslink,
        interface_only=False,
    )
