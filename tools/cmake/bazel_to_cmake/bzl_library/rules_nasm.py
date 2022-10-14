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
"""CMake implementation of "@com_google_tensorstore//:rules_nasm.bzl"."""

# pylint: disable=relative-beyond-top-level,invalid-name,missing-function-docstring,missing-class-docstring

import pathlib
from typing import Optional, List, Set, cast

from ..cmake_builder import CMakeBuilder
from ..cmake_builder import quote_list
from ..cmake_builder import quote_string
from ..configurable import Configurable
from ..evaluation import BazelGlobals
from ..evaluation import Package
from ..evaluation import register_bzl_library
from ..label import CMakeTarget
from ..label import Label
from ..label import RelativeLabel
from ..provider import TargetInfo


@register_bzl_library("@com_google_tensorstore//:rules_nasm.bzl", build=True)
class RulesNasmLibrary(BazelGlobals):

  def bazel_nasm_library(self,
                         name: str,
                         visibility: Optional[List[RelativeLabel]] = None,
                         **kwargs):
    context = self._context
    package = context.current_package
    assert package is not None
    label = package.get_label(name)
    context.add_rule(
        label,
        lambda: _nasm_library_impl(cast(Package, package), label, **kwargs),
        analyze_by_default=package.analyze_by_default(visibility))


def _nasm_library_impl(
    _package: Package,
    _label: Label,
    srcs: Optional[Configurable[List[RelativeLabel]]] = None,
    flags: Optional[Configurable[List[str]]] = None,
    includes: Optional[Configurable[List[RelativeLabel]]] = None,
    alwayslink: bool = False,
    **kwargs,
):
  del kwargs
  context = _package.context
  cmake_target_pair = context.generate_cmake_target_pair(
      _label, generate_alias=True)
  cmake_deps: List[CMakeTarget] = []
  resolved_srcs = context.get_targets_file_paths(
      _package.get_label_list(srcs), cmake_deps)
  resolved_flags = context.evaluate_configurable(flags or [])
  resolved_includes = []
  for include_target in _package.get_label_list(includes or []):
    include_files = context.get_file_paths(include_target, cmake_deps)
    for include_file in include_files:
      resolved_includes.append(str(pathlib.PurePosixPath(include_file).parent))
      # Get include target relative to current package, so that if `includes =
      # ["a/b/whatever.asm"]` then it can be found as either "whatever.asm" or
      # "a/b/whatever.asm".
      package_prefix = _package.repo_and_package_name
      if _package.package_name:
        package_prefix += "/"
      if include_target.startswith(package_prefix):
        relative_target = include_target[len(package_prefix):].replace(":", "/")
        if include_file.endswith(relative_target):
          resolved_includes.append(
              include_file[:-len(relative_target)].rstrip("/"))
  # On Windows, CMake does not correctly handle linking of libraries containing
  # only nasm sources.  As a workaround, add a dummy C file.
  dummy_source: Optional[str] = None
  if context.workspace.cmake_vars["CMAKE_CXX_COMPILER_ID"] == "MSVC":
    dummy_source = _package.context.get_dummy_source()
  _emit_nasm_library(
      context.builder,
      target_name=cmake_target_pair.target,
      alias_name=cast(CMakeTarget, cmake_target_pair.alias),
      cmake_deps=cmake_deps,
      srcs=set(resolved_srcs),
      flags=resolved_flags,
      includes=resolved_includes,
      alwayslink=alwayslink,
      dummy_source=dummy_source,
  )
  context.add_analyzed_target(_label,
                              TargetInfo(*cmake_target_pair.as_providers()))


def _emit_nasm_library(
    _builder: CMakeBuilder,
    target_name: str,
    alias_name: str,
    cmake_deps: List[CMakeTarget],
    srcs: Set[str],
    includes: List[str],
    flags: List[str],
    dummy_source: Optional[str],
    alwayslink: bool,
):
  """Generates an NASM library target."""
  all_srcs = sorted(srcs)
  dummy_sources = [dummy_source] if dummy_source is not None else []
  _builder.addtext(f"""
add_library({target_name})
target_sources({target_name} PRIVATE {quote_list(all_srcs + dummy_sources)})
target_include_directories({target_name} PRIVATE {quote_list(sorted(includes))})
set_source_files_properties(
  {quote_list(all_srcs)}
  PROPERTIES
    LANGUAGE ASM_NASM
    COMPILE_OPTIONS {quote_string(";".join(flags))})
""")
  if cmake_deps:
    _builder.addtext(
        f"add_dependencies({target_name} {quote_list(sorted(cmake_deps))})\n")

  _builder.add_library_alias(
      target_name=target_name,
      alias_name=alias_name,
      alwayslink=alwayslink,
      interface_only=False,
  )
