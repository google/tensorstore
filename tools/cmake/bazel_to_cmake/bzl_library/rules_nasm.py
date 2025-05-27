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
import itertools
import os
import pathlib
from typing import Callable, Collection, List, Optional, cast

from .. import native_rules_cc
from ..cmake_builder import CMakeBuilder
from ..cmake_provider import CMakePackageDepsProvider
from ..cmake_provider import default_providers
from ..cmake_target import CMakeTarget
from ..emit_alias import emit_library_alias
from ..evaluation import EvaluationState
from ..starlark.bazel_target import PackageId
from ..starlark.bazel_target import parse_absolute_target
from ..starlark.bazel_target import TargetId
from ..starlark.common_providers import FilesProvider
from ..starlark.invocation_context import InvocationContext
from ..starlark.invocation_context import RelativeLabel
from ..starlark.provider import TargetInfo
from ..starlark.scope_common import ScopeCommon
from ..starlark.select import Configurable
from ..util import cmake_is_true
from ..util import partition_by
from ..util import quote_list
from ..util import quote_path
from ..util import quote_path_list
from ..util import quote_string
from .register import register_bzl_library


_NASM_SRC_PATTERN = r"\.(?:s|S|asm|nasm)$"

_EMIT_YASM_CHECK = """
get_filename_component(_nasm_compiler_barename "${CMAKE_ASM_NASM_COMPILER}" NAME)
if (_nasm_compiler_barename STREQUAL "yasm")
  message(WARNING "CMake found YASM assembler. Please install 'nasm' instead.")
endif()
unset(_nasm_compiler_barename)

"""

_NASM_TARGET = parse_absolute_target("@nasm//:nasm")


@register_bzl_library("@rules_nasm//nasm:nasm_library.bzl")
class RulesNasmLibrary(ScopeCommon):

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
        lambda: _common_nasm_resolve(
            _nasm_library_impl, context, target, **kwargs
        ),
        visibility=visibility,
    )


@register_bzl_library("@rules_nasm//nasm:nasm_cc_library.bzl")
class RulesNasmCcLibrary(ScopeCommon):

  def bazel_nasm_cc_library(
      self,
      name: str,
      visibility: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    context = self._context.snapshot()
    target = context.resolve_target(name)

    state = context.access(EvaluationState)

    use_cmake_builtin_nasm = True
    if cmake_is_true(
        state.workspace.cmake_vars.get("MSVC_IDE")
    ) or cmake_is_true(state.workspace.cmake_vars.get("XCODE")):
      # The "Xcode" generator does not support built-in nasm at all.
      # The "Visual Studio" generators support nasm but pass "/DWIN32"
      # which is problematic for some packages.
      #
      # NOTE(jbms): https://gitlab.kitware.com/cmake/cmake/-/issues/24079,
      # This rule does not work correctly with the Xcode generator if
      # `alwayslink=True`.
      use_cmake_builtin_nasm = False

    if use_cmake_builtin_nasm:
      context.add_rule(
          target,
          lambda: _common_nasm_resolve(
              _nasm_cc_library_builtin_impl, context, target, **kwargs
          ),
      )
      return

    asm_target = target.get_target_id(f"{target.target_name}__asm")
    context.add_rule(
        asm_target,
        lambda: _common_nasm_resolve(
            _nasm_library_impl, context, asm_target, **kwargs
        ),
    )
    cc_kwargs = kwargs.copy()
    cc_kwargs.pop("srcs", None)
    cc_kwargs.pop("hdrs", None)
    cc_kwargs.pop("preincs", None)
    cc_kwargs.pop("copts", None)
    cc_kwargs.pop("includes", None)
    cc_kwargs["srcs"] = [f":{asm_target.target_name}"]
    native_rules_cc.cc_library(
        self._context,
        name=name,
        visibility=visibility,
        **cc_kwargs,
    )


def _common_nasm_resolve(
    _next: Callable[..., None],
    _context: InvocationContext,
    _target: TargetId,
    srcs: Optional[Configurable[List[RelativeLabel]]] = None,
    hdrs: Optional[Configurable[List[RelativeLabel]]] = None,
    preincs: Optional[Configurable[List[RelativeLabel]]] = None,
    copts: Optional[Configurable[List[str]]] = None,
    includes: Optional[Configurable[List[str]]] = None,
    **kwargs,
):
  """Applies evaluate_configurable to common (and uncommon) nasm arguments."""
  _next(
      _context,
      _target,
      srcs=_context.resolve_target_or_label_list(
          _context.evaluate_configurable_list(srcs)
      ),
      hdrs=_context.resolve_target_or_label_list(
          _context.evaluate_configurable_list(hdrs)
      ),
      preincs=_context.resolve_target_or_label_list(
          _context.evaluate_configurable_list(preincs or [])
      ),
      copts=_context.evaluate_configurable_list(copts or []),
      includes=_context.evaluate_configurable_list(includes or []),
      **kwargs,
  )


def _construct_nasm_includes(
    current_package_id: PackageId,
    source_directory: pathlib.PurePath,
    cmake_binary_dir: pathlib.PurePath,
    includes: Optional[Collection[str]] = None,
):
  """Returns the set of includes for the configuration."""

  raw_includes = {}

  current_package_path = pathlib.PurePosixPath(current_package_id.package_name)

  raw_includes[source_directory] = None
  raw_includes[cmake_binary_dir] = None
  for include in includes or []:
    constructed = str(current_package_path.joinpath(include))
    if constructed[0] == "/":
      constructed = constructed[1:]
    raw_includes[source_directory.joinpath(constructed)] = None
    raw_includes[cmake_binary_dir.joinpath(constructed)] = None

  # NOTE: Remove from here and @rules_nasm
  raw_includes[source_directory.joinpath(current_package_path)] = None
  return raw_includes.keys()


def _get_obj_suffix(state: EvaluationState):
  o_ext = state.workspace.cmake_vars.get("CMAKE_C_OUTPUT_EXTENSION", None)
  if o_ext is not None:
    return o_ext
  cmake_system_name = state.workspace.cmake_vars["CMAKE_SYSTEM_NAME"]
  if cmake_system_name == "Windows":
    return ".obj"
  else:
    return ".o"


def _nasm_library_impl(
    _context: InvocationContext,
    _target: TargetId,
    *,
    srcs: List[TargetId],
    hdrs: List[TargetId],
    preincs: List[TargetId],
    copts: List[str],
    includes: List[str],
    **kwargs,
):
  del kwargs
  state = _context.access(EvaluationState)
  repo = state.workspace.all_repositories.get(
      _context.caller_package_id.repository_id
  )
  assert repo is not None

  srcs_collector = state.collect_targets(srcs)
  preincs_collector = state.collect_targets(preincs)

  cmake_deps: List[CMakeTarget] = list(
      itertools.chain(
          srcs_collector.add_dependencies(),
          preincs_collector.add_dependencies(),
          state.collect_targets(hdrs).add_dependencies(),
      )
  )

  # Maybe add a dependency on @nasm//:nasm
  maybe_nasm = state.get_optional_target_info(_NASM_TARGET)
  if maybe_nasm is not None:
    p = maybe_nasm.get(CMakePackageDepsProvider)
    if p is not None:
      cmake_deps.append(p.package)

  copts_with_preinc = []
  for inc in _construct_nasm_includes(
      _target.package_id,
      repo.source_directory,
      repo.cmake_binary_dir,
      includes,
  ):
    copts_with_preinc.append(f"-I{str(inc)}")
  for preinc in preincs_collector.file_paths():
    copts_with_preinc.append(f"-p{str(preinc)}")
  copts_with_preinc.extend(copts)

  o_ext = _get_obj_suffix(state)

  def _output_file(src: pathlib.PurePath):
    return os.path.join(
        repo.cmake_binary_dir,
        "_nasm",
        _target.target_name,
        hashlib.sha256(
            (_target.as_label() + str(src)).encode("utf-8")
        ).hexdigest()[:8],
        os.path.basename(src) + o_ext,
    )

  generated_files = []

  # Emit
  out = io.StringIO()
  out.write(f"\n# nasm_library({_target.as_label()})\n")

  for src in srcs_collector.file_paths():
    generated_path = _output_file(src)
    generated_files.append(generated_path)
    _emit_nasm_assemble(
        out,
        src,
        generated_path,
        copts_with_preinc,
        cmake_deps,
    )

  builder = _context.access(CMakeBuilder)
  builder.addtext(_EMIT_YASM_CHECK, unique=True)
  builder.addtext(out.getvalue())

  cmake_target_pair = state.generate_cmake_target_pair(_target)
  _context.add_analyzed_target(
      _target,
      TargetInfo(
          CMakePackageDepsProvider(cmake_target_pair.cmake_package),
          FilesProvider(generated_files),
      ),
  )


def _emit_nasm_assemble(
    out: io.StringIO,
    src: pathlib.PurePath,
    generated_obj: pathlib.PurePath,
    copts: List[str],
    cmake_deps: List[CMakeTarget],
):
  """Generates an NASM library target."""
  out.write(f"""add_custom_command(
  OUTPUT {quote_path(generated_obj)}
  DEPENDS {quote_path_list([src] + cast(List[str], cmake_deps))}
  COMMAND ${{CMAKE_ASM_NASM_COMPILER}}
          -f ${{CMAKE_ASM_NASM_OBJECT_FORMAT}}
          ${{CMAKE_ASM_NASM_FLAGS}}
          {quote_list(copts)}
          -o {quote_path(generated_obj)}
          {quote_path(src)}
  COMMENT "Assembling NASM source {src}"
)
set_source_files_properties({quote_path(generated_obj)}
  PROPERTIES GENERATED TRUE)
""")


# Native cmake NASM library.
def _nasm_cc_library_builtin_impl(
    _context: InvocationContext,
    _target: TargetId,
    *,
    srcs: List[TargetId],
    hdrs: List[TargetId],
    preincs: List[TargetId],
    copts: List[str],
    includes: List[str],
    alwayslink: bool = False,
    **kwargs,
):
  del kwargs
  state = _context.access(EvaluationState)
  repo = state.workspace.all_repositories.get(
      _context.caller_package_id.repository_id
  )
  assert repo is not None

  srcs_collector = state.collect_targets(srcs)
  preincs_collector = state.collect_targets(preincs)

  cmake_deps: List[CMakeTarget] = list(
      itertools.chain(
          srcs_collector.add_dependencies(),
          preincs_collector.add_dependencies(),
          state.collect_targets(hdrs).add_dependencies(),
      )
  )

  # Maybe add a dependency on @nasm//:nasm
  maybe_nasm = state.get_optional_target_info(_NASM_TARGET)
  if maybe_nasm is not None:
    p = maybe_nasm.get(CMakePackageDepsProvider)
    if p is not None:
      cmake_deps.append(p.package)

  nasm_includes = _construct_nasm_includes(
      _target.package_id,
      repo.source_directory,
      repo.cmake_binary_dir,
      includes,
  )
  copts_with_preinc = []
  for preinc in preincs_collector.file_paths():
    copts_with_preinc.append(f"-p{str(preinc)}")
  copts_with_preinc.extend(copts)

  nasm_srcs = list(srcs_collector.file_paths())

  # On Windows, CMake does not correctly handle linking of libraries containing
  # only nasm sources.
  if state.workspace.cmake_vars["CMAKE_CXX_COMPILER_ID"] == "MSVC":
    nasm_srcs.append(state.get_placeholder_source())

  cmake_target_pair = state.generate_cmake_target_pair(_target)

  # Emit
  out = io.StringIO()
  out.write(f"\n# nasm_cc_library({_target.as_label()})\n")
  _emit_nasm_library(
      out,
      target_name=cmake_target_pair.target,
      srcs=nasm_srcs,
      includes=nasm_includes,
      copts=copts_with_preinc,
      cmake_deps=cmake_deps,
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
    srcs: set[pathlib.PurePath],
    includes: List[pathlib.PurePath],
    copts: List[str],
    cmake_deps: List[CMakeTarget],
):
  """Generates an NASM library target."""
  _sep = "\n    "

  sorted_srcs = sorted(srcs)
  asm_srcs = partition_by(sorted_srcs, pattern=_NASM_SRC_PATTERN)[0]

  out.write(f"add_library({target_name})\n")
  out.write(
      f"""target_sources({target_name} PRIVATE{_sep}{quote_path_list(sorted_srcs, separator=_sep)})
target_include_directories({target_name} PRIVATE{_sep}{quote_path_list(sorted(includes), separator=_sep)})
set_source_files_properties({_sep}{quote_path_list(asm_srcs, separator=_sep)}
    PROPERTIES
      LANGUAGE ASM_NASM
      COMPILE_OPTIONS {quote_string(";".join(copts))})\n"""
  )
  if cmake_deps:
    out.write(
        f"add_dependencies({target_name} {quote_list(sorted(cmake_deps))})\n"
    )
