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
from .package import Package
from .package import Visibility
from .protoc_helper import protoc_compile_protos_impl
from .starlark.bazel_globals import register_native_build_rule
from .starlark.bazel_target import TargetId
from .starlark.common_providers import ProtoLibraryProvider
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.provider import TargetInfo
from .starlark.select import Configurable
from .variable_substitution import apply_location_substitutions

_SEP = "\n        "


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

  include_dirs = ["${PROJECT_SOURCE_DIR}", "${PROJECT_BINARY_DIR}"]
  if includes is not None:
    include_dirs.extend(includes)
  include_dirs = [
      f"$<BUILD_INTERFACE:{include_dir}>" for include_dir in include_dirs
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


def _handle_cc_common_options(
    _context: InvocationContext,
    src_required=False,
    custom_target_deps: Optional[List[CMakeTarget]] = None,
    srcs: Optional[Configurable[List[RelativeLabel]]] = None,
    deps: Optional[Configurable[List[RelativeLabel]]] = None,
    includes: Optional[Configurable[List[str]]] = None,
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

  include_dirs = _context.evaluate_configurable_list(includes)

  if strip_include_prefix is not None:
    include_dirs.append(strip_include_prefix)

  package = _context.access(Package)
  resolved_includes: List[str] = []
  for include in include_dirs:
    resolved_includes.append(
        str(
            pathlib.PurePosixPath(package.repository.source_directory).joinpath(
                package.package_id.package_name, include)))
    resolved_includes.append(
        str(
            pathlib.PurePosixPath(package.repository.cmake_binary_dir).joinpath(
                package.package_id.package_name, include)))
  result["includes"] = sorted(set(resolved_includes))
  return result


@register_native_build_rule
def cc_library(self: InvocationContext,
               name: str,
               visibility: Optional[List[RelativeLabel]] = None,
               **kwargs):
  context = self.snapshot()
  target = context.resolve_target(name)
  context.add_rule(
      target,
      lambda: _cc_library_impl(context, target, **kwargs),
      visibility=visibility)


def _cc_library_impl(
    _context: InvocationContext,
    _target: TargetId,
    hdrs: Optional[Configurable[List[RelativeLabel]]] = None,
    textual_hdrs: Optional[Configurable[List[RelativeLabel]]] = None,
    alwayslink: bool = False,
    **kwargs,
):
  resolved_hdrs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(hdrs))
  resolved_textual_hdrs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(textual_hdrs))

  state = _context.access(EvaluationState)

  cmake_target_pair = state.generate_cmake_target_pair(
      _target, generate_alias=True)
  custom_target_deps: List[CMakeTarget] = []
  hdrs_file_paths = state.get_targets_file_paths(
      resolved_hdrs, custom_target_deps=custom_target_deps)
  textual_hdrs_file_paths = state.get_targets_file_paths(
      resolved_textual_hdrs, custom_target_deps=custom_target_deps)
  emit_cc_library(
      _context.access(CMakeBuilder),
      cmake_target_pair,
      hdrs=set(hdrs_file_paths + textual_hdrs_file_paths),
      alwayslink=alwayslink,
      **_handle_cc_common_options(
          _context, custom_target_deps=custom_target_deps, **kwargs),
  )
  _context.add_analyzed_target(_target,
                               TargetInfo(*cmake_target_pair.as_providers()))


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
  assert _cmake_target_pair.alias is not None

  if not header_only:
    _builder.addtext(f"""\n
add_library({target_name})
target_sources({target_name} PRIVATE{_SEP}{quote_list(cc_srcs , separator=_SEP)})
set_property(TARGET {target_name} PROPERTY LINKER_LANGUAGE "CXX")
""")
  else:
    _builder.addtext(f"""\n
add_library({target_name} INTERFACE)
""")
  _emit_cc_common_options(
      _builder, target_name=target_name, interface_only=header_only, **kwargs)
  _builder.add_library_alias(
      target_name=target_name,
      alias_name=_cmake_target_pair.alias,
      alwayslink=alwayslink,
      interface_only=header_only,
  )


@register_native_build_rule
def cc_binary(self: InvocationContext,
              name: str,
              visibility: Optional[List[RelativeLabel]] = None,
              **kwargs):
  context = self.snapshot()
  target = context.resolve_target(name)

  resolved_visibility = context.resolve_target_or_label_list(visibility or [])
  if kwargs.get("testonly"):
    analyze_by_default = context.access(Visibility).analyze_test_by_default(
        resolved_visibility)
  else:
    analyze_by_default = context.access(Visibility).analyze_by_default(
        resolved_visibility)

  context.add_rule(
      target,
      lambda: _cc_binary_impl(context, target, **kwargs),
      analyze_by_default=analyze_by_default)


def _cc_binary_impl(_context: InvocationContext, _target: TargetId, **kwargs):
  cmake_target_pair = _context.access(
      EvaluationState).generate_cmake_target_pair(
          _target, generate_alias=True)
  _emit_cc_binary(
      _context.access(CMakeBuilder),
      cmake_target_pair,
      **_handle_cc_common_options(_context, src_required=True, **kwargs),
  )
  _context.add_analyzed_target(_target,
                               TargetInfo(*cmake_target_pair.as_providers()))


def _emit_cc_binary(_builder: CMakeBuilder, _cmake_target_pair: CMakeTargetPair,
                    srcs: Set[str], **kwargs):
  target_name = _cmake_target_pair.target
  assert _cmake_target_pair.alias is not None
  _builder.addtext(f"""\n
add_executable({target_name} "")
add_executable({_cmake_target_pair.alias} ALIAS {target_name})
target_sources({target_name} PRIVATE{_SEP}{quote_list(sorted(srcs), separator=_SEP)})
""")
  _emit_cc_common_options(_builder, target_name=target_name, **kwargs)


@register_native_build_rule
def cc_test(self: InvocationContext,
            name: str,
            visibility: Optional[List[RelativeLabel]] = None,
            **kwargs):
  context = self.snapshot()
  target = context.resolve_target(name)

  resolved_visibility = context.resolve_target_or_label_list(visibility or [])
  context.add_rule(
      target,
      lambda: _cc_test_impl(context, target, **kwargs),
      analyze_by_default=context.access(Visibility).analyze_test_by_default(
          resolved_visibility))


def _cc_test_impl(_context: InvocationContext,
                  _target: TargetId,
                  args: Optional[Configurable[List[str]]] = None,
                  **kwargs):
  state = _context.access(EvaluationState)
  cmake_target_pair = state.generate_cmake_target_pair(
      _target, generate_alias=True)
  resolved_args = [
      apply_location_substitutions(
          _context, arg, relative_to=state.repo.source_directory)
      for arg in _context.evaluate_configurable_list(args)
  ]
  _emit_cc_test(
      _context.access(CMakeBuilder),
      cmake_target_pair,
      args=resolved_args,
      **_handle_cc_common_options(_context, src_required=True, **kwargs),
  )
  _context.add_analyzed_target(_target,
                               TargetInfo(*cmake_target_pair.as_providers()))


def _emit_cc_test(_builder: CMakeBuilder,
                  _cmake_target_pair: CMakeTargetPair,
                  args: Optional[List[str]] = None,
                  **kwargs):
  _emit_cc_binary(_builder, _cmake_target_pair, **kwargs)
  target_name = _cmake_target_pair.target
  args_suffix = ""
  if args:
    args_suffix = " " + " ".join(args)
  _builder.addtext(
      f"""add_test(NAME {target_name} COMMAND {target_name}{args_suffix} WORKING_DIRECTORY ${{CMAKE_CURRENT_SOURCE_DIR}})\n"""
  )


@register_native_build_rule
def cc_proto_library(self: InvocationContext,
                     name: str,
                     visibility: Optional[List[RelativeLabel]] = None,
                     **kwargs):
  context = self.snapshot()
  target = context.resolve_target(name)
  context.add_rule(
      target,
      lambda: _cc_proto_library_impl(context, target, **kwargs),
      visibility=visibility)


def _cc_proto_library_impl(_context: InvocationContext,
                           _label: TargetId,
                           deps: Optional[List[RelativeLabel]] = None,
                           **kwargs):
  del kwargs
  resolved_deps = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(deps))

  state = _context.access(EvaluationState)
  cmake_target_pair = state.generate_cmake_target_pair(
      _label, generate_alias=True)

  dep_library_targets = [
      _get_cc_proto_library_target(_context, dep_target)
      for dep_target in resolved_deps
  ]
  emit_cc_library(
      _context.access(CMakeBuilder),
      cmake_target_pair,
      hdrs=set(),
      srcs=set(),
      deps=set(state.get_deps(dep_library_targets)),
  )
  _context.add_analyzed_target(_label,
                               TargetInfo(*cmake_target_pair.as_providers()))


def _get_cc_proto_library_target(context: InvocationContext,
                                 proto_target: TargetId) -> TargetId:
  proto_info = context.get_target_info(proto_target).get(ProtoLibraryProvider)
  if proto_info is None:
    # This could be external; defer to get_deps.
    return proto_target

  cc_deps: List[TargetId] = [
      _get_single_cc_proto_target(context, proto_src, proto_info.deps)
      for proto_src in proto_info.srcs
  ]
  if len(cc_deps) == 1:
    return cc_deps[0]

  cc_library_target = proto_target.get_target_id(
      f"{proto_target.target_name}__cc_proto_library")

  state = context.access(EvaluationState)
  cmake_target_pair = state.generate_cmake_target_pair(
      cc_library_target, generate_alias=True)

  emit_cc_library(
      context.access(CMakeBuilder),
      cmake_target_pair,
      hdrs=set(),
      srcs=set(),
      deps=set(state.get_deps(cc_deps)),
  )
  return cc_library_target


def _get_single_cc_proto_target(context: InvocationContext, proto_src: TargetId,
                                deps: List[TargetId]) -> TargetId:

  cc_library_target = proto_src.get_target_id(proto_src.target_name +
                                              "__cc_proto")
  protoc_output_target = proto_src.get_target_id(proto_src.target_name +
                                                 "__cc_protoc")

  state = context.access(EvaluationState)
  info = state.get_optional_target_info(cc_library_target)
  if info is not None:
    return cc_library_target

  generated = protoc_compile_protos_impl(
      context,
      protoc_output_target,
      proto_src,
      plugin=None,
      add_files_provider=False)

  cmake_cc_target_pair = state.generate_cmake_target_pair(
      cc_library_target, generate_alias=True)

  context.add_analyzed_target(cc_library_target,
                              TargetInfo(*cmake_cc_target_pair.as_providers()))

  cc_deps: List[TargetId] = [
      _get_cc_proto_library_target(context, proto_dep) for proto_dep in deps
  ]
  cc_deps.append(proto_src.parse_target("@com_google_protobuf//:protobuf"))

  cmake_cc_deps = state.get_deps(cc_deps)
  builder = context.access(CMakeBuilder)

  emit_cc_library(
      builder,
      cmake_cc_target_pair,
      hdrs=set(filter(lambda x: x.endswith(".h"), generated.paths)),
      srcs=set(filter(lambda x: not x.endswith(".h"), generated.paths)),
      deps=set(cmake_cc_deps),
  )
  builder.addtext(
      f"add_dependencies({cmake_cc_target_pair.target} {state.get_dep(protoc_output_target)[0]})\n"
  )
  return cc_library_target
