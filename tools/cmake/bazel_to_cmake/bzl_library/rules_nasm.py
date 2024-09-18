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
"""CMake implementation of "@tensorstore//bazel:rules_nasm.bzl"."""

# pylint: disable=relative-beyond-top-level,invalid-name,missing-function-docstring,missing-class-docstring

import hashlib
import io
import os
import pathlib
from typing import Iterable, List, Optional, cast

from ..cmake_builder import CMakeBuilder
from ..cmake_provider import default_providers
from ..cmake_target import CMakeTarget
from ..emit_alias import emit_library_alias
from ..evaluation import EvaluationState
from ..starlark.bazel_globals import BazelGlobals
from ..starlark.bazel_globals import register_bzl_library
from ..starlark.bazel_target import TargetId
from ..starlark.invocation_context import InvocationContext
from ..starlark.invocation_context import RelativeLabel
from ..starlark.provider import TargetInfo
from ..starlark.select import Configurable
from ..util import cmake_is_true
from ..util import quote_list
from ..util import quote_path
from ..util import quote_path_list
from ..util import quote_string


_EMIT_YASM_CHECK = """
get_filename_component(_nasm_compiler_barename "${CMAKE_ASM_NASM_COMPILER}" NAME)
if (_nasm_compiler_barename STREQUAL "yasm")
  message(WARNING "CMake found YASM assembler. Please install 'nasm' instead.")
endif()
unset(_nasm_compiler_barename)

"""


@register_bzl_library("@tensorstore//bazel:rules_nasm.bzl", build=True)
class RulesNasmLibrary(BazelGlobals):

  def bazel_nasm_library(
      self,
      name: str,
      visibility: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    context = self._context.snapshot()

    target = context.resolve_target(name)
    context.add_rule(
        target,
        lambda: _nasm_library_impl(context, target, **kwargs),
        visibility=visibility,
    )


def _nasm_includes(
    _context: InvocationContext,
    resolved_includes: Iterable[TargetId],
    cmake_deps: List[CMakeTarget],
) -> set[str]:
  state = _context.access(EvaluationState)

  # Construct a package prefix string to compare against the include target.
  package_prefix_str = _context.caller_repository_name
  if _context.caller_package_id.package_name:
    package_prefix_str += "/"

  all_includes: set[str] = set()
  for include_target in resolved_includes:
    # Get include target relative to current package, so that if `includes =
    # ["a/b/whatever.asm"]` then it can be found as either "whatever.asm" or
    # "a/b/whatever.asm".
    relative_target = None
    include_target_str = include_target.as_label()
    if include_target_str.startswith(package_prefix_str):
      relative_target = include_target_str[len(package_prefix_str) :].replace(
          ":", "/"
      )
    collector = state.collect_targets([include_target])
    cmake_deps.extend(collector.add_dependencies())

    for include_file in collector.file_paths():
      all_includes.add(str(pathlib.PurePosixPath(include_file).parent))
      if relative_target and include_file.endswith(relative_target):
        all_includes.add(include_file[: -len(relative_target)].rstrip("/"))

  return all_includes


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

  cmake_target_pair = state.generate_cmake_target_pair(_target)
  cmake_deps: List[CMakeTarget] = []

  resolved_flags = _context.evaluate_configurable(flags or [])

  resolved_srcs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(srcs or [])
  )
  resolved_includes = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(includes or [])
  )

  srcs_collector = state.collect_targets(resolved_srcs)

  use_builtin_rule = True

  workspace = state.workspace
  if cmake_is_true(workspace.cmake_vars.get("MSVC_IDE")) or cmake_is_true(
      workspace.cmake_vars.get("XCODE")
  ):
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
  # add a placeholder C file.
  placeholder_source: Optional[str] = None
  if (
      workspace.cmake_vars["CMAKE_CXX_COMPILER_ID"] == "MSVC"
      or not use_builtin_rule
  ):
    placeholder_source = state.get_placeholder_source()

  # Emit
  out = io.StringIO()
  out.write(f"\n# nasm_library({_target.as_label()})\n")
  _emit_nasm_library(
      out,
      target_name=cmake_target_pair.target,
      cmake_deps=cmake_deps,
      srcs=set(srcs_collector.file_paths()),
      flags=resolved_flags,
      includes=_nasm_includes(_context, resolved_includes, cmake_deps),
      placeholder_source=placeholder_source,
      use_builtin_rule=use_builtin_rule,
  )
  extra_providers = ()
  if cmake_target_pair.alias:
    extra_providers = emit_library_alias(
        out,
        target_name=cmake_target_pair.target,
        alias_name=cmake_target_pair.alias,
        alwayslink=alwayslink,
        interface_only=False,
    )

  builder = _context.access(CMakeBuilder)
  builder.addtext(_EMIT_YASM_CHECK, unique=True)
  builder.addtext(out.getvalue())

  _context.add_analyzed_target(
      _target,
      TargetInfo(*default_providers(cmake_target_pair), *extra_providers),
  )


def _emit_nasm_library(
    out: io.StringIO,
    target_name: CMakeTarget,
    cmake_deps: List[CMakeTarget],
    srcs: set[str],
    includes: set[str],
    flags: List[str],
    placeholder_source: Optional[str],
    use_builtin_rule: bool,
):
  """Generates an NASM library target."""
  all_srcs = sorted(srcs)
  placeholder_sources = (
      [placeholder_source] if placeholder_source is not None else []
  )

  _sep = "\n    "

  out.write(f"add_library({target_name})\n")
  if use_builtin_rule:
    out.write(
        f"""target_sources({target_name} PRIVATE {quote_list(all_srcs + placeholder_sources, separator=_sep)})
target_include_directories({target_name} PRIVATE {quote_list(sorted(includes), separator=_sep)})
set_source_files_properties(
    {quote_list(all_srcs, separator=_sep)}
    PROPERTIES
      LANGUAGE ASM_NASM
      COMPILE_OPTIONS {quote_string(";".join(flags))})\n"""
    )
    if cmake_deps:
      out.write(
          f"add_dependencies({target_name} {quote_list(sorted(cmake_deps))})\n"
      )
  else:
    all_flags = list(flags)
    for include in includes:
      all_flags.append("-I" + include)
    all_src_obj_exprs = []
    for src in sorted(srcs):
      src_obj_base = (
          os.path.basename(src)
          + "_"
          + hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
      )
      src_obj_expr = f"${{CMAKE_CURRENT_BINARY_DIR}}/{src_obj_base}${{CMAKE_C_OUTPUT_EXTENSION}}"
      all_src_obj_exprs.append(src_obj_expr)
      out.write(f"""
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
    out.write(
        f"target_sources({target_name} PRIVATE\n"
        f"{_sep}{quote_list(all_src_obj_exprs + placeholder_sources, separator=_sep)})\n"
    )
