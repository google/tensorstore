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
"""CMake implementation of native Bazel build rules."""

# pylint: disable=relative-beyond-top-level,invalid-name,missing-function-docstring,g-long-lambda

import glob as _glob
import os
import pathlib
import re
from typing import List, Optional, Set, Dict, Any, cast

from . import cmake_builder
from .cmake_builder import CMakeBuilder
from .cmake_builder import quote_list
from .configurable import Configurable
from .evaluation import EvaluationContext
from .evaluation import Package
from .evaluation import register_native_build_rule
from .label import CMakeTarget
from .label import Label
from .label import RelativeLabel
from .provider import CMakeDepsProvider
from .provider import CMakeTargetPair
from .provider import ConditionProvider
from .provider import FilesProvider
from .provider import Provider
from .provider import TargetInfo
from .util import glob_pattern_to_regexp
from .variable_substitution import apply_location_substitutions
from .variable_substitution import apply_make_variable_substitutions


@register_native_build_rule
def glob(self: EvaluationContext,
         include: List[str],
         exclude: Optional[List[str]] = None,
         allow_empty: bool = True) -> List[str]:
  matches = set()

  exclude_regexp = None
  if exclude:
    exclude_regexp = re.compile("(?:" + "|".join(
        glob_pattern_to_regexp(pattern) for pattern in exclude) + ")")

  assert self.current_package_name is not None
  package_directory = str(
      pathlib.PurePosixPath(self.repo.source_directory).joinpath(
          self.current_package_name))

  is_subpackage_dir: Dict[str, bool] = {}
  is_subpackage_dir[""] = False
  def in_subpackage(path: str):
    start_index = 1
    end_index = len(path)
    while start_index < end_index:
      index = path.find("/", start_index)
      if index == -1:
        break
      start_index = index + 1
      subdir = path[:index]
      result = is_subpackage_dir.get(subdir)
      if result is not None:
        return result
      build_path = str(
          pathlib.PurePosixPath(package_directory).joinpath(subdir, "BUILD"))
      result = (
          os.path.exists(build_path) or os.path.exists(build_path + ".bazel"))
      is_subpackage_dir[subdir] = result
      if result:
        return True
    return False

  def get_matches(pattern: str):
    for match in _glob.iglob(
        os.path.join(package_directory, pattern), recursive=True):
      if not os.path.isfile(match):
        continue
      relative = os.path.relpath(match, package_directory)
      if os.sep != "/":
        relative = relative.replace(os.sep, "/")
      if in_subpackage(relative):
        continue
      if exclude_regexp is not None and exclude_regexp.fullmatch(relative):
        continue
      yield relative

  for pattern in include:
    matches.update(get_matches(pattern))

  if not matches and not allow_empty:
    raise ValueError("glob produced empty result list")
  return sorted(matches)


@register_native_build_rule
def package_name(self: EvaluationContext):
  return self.current_package_name


@register_native_build_rule
def repository_name(self: EvaluationContext):
  return self.current_repository_name


@register_native_build_rule
def package_group(self: EvaluationContext, **kwargs):
  del self
  del kwargs
  pass


@register_native_build_rule
def filegroup(self: EvaluationContext, name: str, srcs: List[RelativeLabel],
              **kwargs):
  del kwargs
  package = self.current_package
  assert package is not None
  target = package.get_label(name)

  def impl():
    cmake_deps: List[CMakeTarget] = []
    providers = [
        FilesProvider(
            package.context.get_targets_file_paths(
                package.get_label_list(srcs), cmake_deps))
    ]
    if cmake_deps:
      providers.append(CMakeDepsProvider(cmake_deps))
    self.add_analyzed_target(target, TargetInfo(*providers))

  self.add_rule(target, impl, analyze_by_default=False)


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
  public_scope = "INTERFACE" if interface_only else "PUBLIC"
  if local_defines and not interface_only:
    _builder.addtext(
        f"target_compile_definitions({target_name} PRIVATE {quote_list(local_defines)})\n"
    )
  if defines:
    _builder.addtext(
        f"target_compile_definitions({target_name} {public_scope} {quote_list(defines)})\n"
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
        f"target_link_libraries({target_name} {public_scope} {quote_list(link_libs)})\n"
    )
  _builder.addtext(
      f"target_include_directories({target_name} {public_scope} {quote_list(include_dirs)})\n"
  )
  _builder.addtext(
      f"target_compile_features({target_name} {public_scope} cxx_std_17)\n")
  if custom_target_deps:
    _builder.addtext(
        f"add_dependencies({target_name} {quote_list(custom_target_deps)})\n")
  if extra_public_compile_options:
    _builder.addtext(
        f"target_compile_options({target_name} {public_scope} {quote_list(extra_public_compile_options)})\n"
    )


def _handle_cc_common_options(
    _package: Package,
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
  resolved_srcs = _package.context.get_targets_file_paths(
      _package.get_label_list(srcs), custom_target_deps)
  if src_required and not resolved_srcs:
    resolved_srcs = [_package.context.get_dummy_source()]
  cmake_deps = set(_package.context.get_deps(_package.get_label_list(deps)))

  # Since Bazel implicitly adds a dependency on the C math library, also add
  # it here.
  if _package.repo.workspace.cmake_vars["CMAKE_SYSTEM_NAME"] != "Windows":
    cmake_deps.add("m")

  cmake_deps.add("Threads::Threads")

  extra_public_compile_options = []

  def add_compile_options(lang: str, options: List[str]):
    for option in options:
      extra_public_compile_options.append(
          f"$<$<COMPILE_LANGUAGE:{lang}>:{option}>")

  add_compile_options("C,CXX", _package.repo.workspace.copts)
  add_compile_options("CXX", _package.repo.workspace.cxxopts)

  result: Dict[str, Any] = {
      "srcs": set(resolved_srcs),
      "deps": cmake_deps,
      "custom_target_deps": set(custom_target_deps),
      "extra_public_compile_options": extra_public_compile_options,
  }
  for k in ["copts", "linkopts", "defines", "local_defines"]:
    value = kwargs.get(k)
    if value is None:
      value = []
    result[k] = _package.context.evaluate_configurable(cast(Any, value))

  result["defines"].extend(_package.repo.workspace.cdefines)

  if includes is None:
    include_dirs = []
  else:
    include_dirs = _package.context.evaluate_configurable(includes)

  if strip_include_prefix is not None:
    include_dirs.append(strip_include_prefix)

  resolved_includes: List[str] = []
  for include in include_dirs:
    resolved_includes.append(
        str(
            pathlib.PurePosixPath(_package.repo.source_directory).joinpath(
                _package.package_name, include)))
    resolved_includes.append(
        str(
            pathlib.PurePosixPath(_package.repo.cmake_binary_dir).joinpath(
                _package.package_name, include)))
  result["includes"] = resolved_includes
  return result


@register_native_build_rule
def cc_library(self: EvaluationContext,
               name: str,
               visibility: Optional[List[RelativeLabel]] = None,
               **kwargs):
  package = self.current_package
  assert package is not None
  target = package.get_label(name)
  self.add_rule(
      target,
      lambda: _cc_library_impl(cast(Package, package), target, **kwargs),
      analyze_by_default=package.analyze_by_default(visibility))


def _cc_library_impl(
    _package: Package,
    _target: Label,
    hdrs: Optional[Configurable[List[RelativeLabel]]] = None,
    textual_hdrs: Optional[Configurable[List[RelativeLabel]]] = None,
    alwayslink: bool = False,
    **kwargs,
):
  context = _package.context
  cmake_target_pair = _package.context.generate_cmake_target_pair(
      _target, generate_alias=True)
  custom_target_deps: List[CMakeTarget] = []
  resolved_hdrs = context.get_targets_file_paths(
      _package.get_label_list(hdrs), custom_target_deps=custom_target_deps)
  resolved_textual_hdrs = context.get_targets_file_paths(
      _package.get_label_list(textual_hdrs),
      custom_target_deps=custom_target_deps)
  _emit_cc_library(
      context.builder,
      cmake_target_pair,
      hdrs=set(resolved_hdrs + resolved_textual_hdrs),
      alwayslink=alwayslink,
      **_handle_cc_common_options(
          _package, custom_target_deps=custom_target_deps, **kwargs),
  )
  context.add_analyzed_target(_target,
                              TargetInfo(*cmake_target_pair.as_providers()))


def _emit_cc_library(
    _builder: CMakeBuilder,
    _cmake_target_pair: CMakeTargetPair,
    srcs: Set[str],
    hdrs: Set[str],
    alwayslink: bool = False,
    **kwargs,
):
  """Generates a C++ library target."""
  cc_srcs = [x for x in srcs if not re.search(r"\.(?:h|inc)$", x)]
  header_only = not cc_srcs
  del hdrs

  target_name = _cmake_target_pair.target
  assert target_name is not None
  alias_name = _cmake_target_pair.alias
  assert alias_name is not None
  if not header_only:
    _builder.addtext(f"""\n
add_library({target_name})
target_sources({target_name} PRIVATE {quote_list(cc_srcs)})
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
      alias_name=alias_name,
      alwayslink=alwayslink,
      interface_only=header_only,
  )


@register_native_build_rule
def cc_binary(self: EvaluationContext,
              name: str,
              visibility: Optional[List[RelativeLabel]] = None,
              **kwargs):
  package = self.current_package
  assert package is not None
  target = package.get_label(name)
  self.add_rule(
      target,
      lambda: _cc_binary_impl(cast(Package, package), target, **kwargs),
      analyze_by_default=package.analyze_by_default(visibility))


def _cc_binary_impl(_package: Package, _target: Label, **kwargs):
  cmake_target_pair = _package.context.generate_cmake_target_pair(
      _target, generate_alias=True)
  _emit_cc_binary(
      _package.context.builder,
      cmake_target_pair,
      **_handle_cc_common_options(_package, src_required=True, **kwargs),
  )
  _package.context.add_analyzed_target(
      _target, TargetInfo(*cmake_target_pair.as_providers()))


def _emit_cc_binary(_builder: CMakeBuilder, _cmake_target_pair: CMakeTargetPair,
                    srcs: Set[str], **kwargs):
  target_name = _cmake_target_pair.target
  alias_name = cast(CMakeTarget, _cmake_target_pair.alias)
  _builder.addtext(f"""\n
add_executable({target_name} "")
add_executable({alias_name} ALIAS {target_name})
target_sources({target_name} PRIVATE {quote_list(srcs)})
""")
  _emit_cc_common_options(_builder, target_name=target_name, **kwargs)


@register_native_build_rule
def cc_test(self: EvaluationContext,
            name: str,
            visibility: Optional[List[RelativeLabel]] = None,
            **kwargs):
  package = self.current_package
  assert package is not None
  target = package.get_label(name)

  self.add_rule(
      target,
      lambda: _cc_test_impl(cast(Package, package), target, **kwargs),
      analyze_by_default=package.analyze_test_by_default(visibility))


def _cc_test_impl(_package: Package,
                  _target: Label,
                  args: Optional[Configurable[List[str]]] = None,
                  **kwargs):
  cmake_target_pair = _package.context.generate_cmake_target_pair(
      _target, generate_alias=True)
  resolved_args = []
  if args:
    resolved_args = _package.context.evaluate_configurable(args)
    resolved_args = [
        apply_location_substitutions(
            _package, arg, relative_to=_package.repo.source_directory)
        for arg in resolved_args
    ]
  _emit_cc_test(
      _package.context.builder,
      cmake_target_pair,
      args=resolved_args,
      **_handle_cc_common_options(_package, src_required=True, **kwargs),
  )
  _package.context.add_analyzed_target(
      _target, TargetInfo(*cmake_target_pair.as_providers()))


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


class ProtoLibraryProvider(Provider):
  __slots__ = ("srcs", "deps")

  def __init__(self, srcs: List[Label], deps: List[Label]):
    self.srcs = set(srcs)
    self.deps = set(deps)


@register_native_build_rule
def proto_library(self: EvaluationContext,
                  name: str,
                  visibility: Optional[List[RelativeLabel]] = None,
                  **kwargs):
  package = self.current_package
  assert package is not None
  label = package.get_label(name)
  self.add_rule(
      label,
      lambda: _proto_library_impl(cast(Package, package), label, **kwargs),
      analyze_by_default=package.analyze_by_default(visibility))


def _proto_library_impl(
    _package: Package,
    _label: Label,
    srcs: Optional[List[RelativeLabel]] = None,
    deps: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  del kwargs
  resolved_srcs = _package.get_label_list(srcs)
  resolved_deps = _package.get_label_list(deps)
  _package.context.add_analyzed_target(
      _label, TargetInfo(ProtoLibraryProvider(resolved_srcs, resolved_deps)))


@register_native_build_rule
def cc_proto_library(self: EvaluationContext,
                     name: str,
                     visibility: Optional[List[RelativeLabel]] = None,
                     **kwargs):
  package = self.current_package
  assert package is not None
  label = package.get_label(name)
  self.add_rule(
      label,
      lambda: _cc_proto_library_impl(cast(Package, package), label, **kwargs),
      analyze_by_default=package.analyze_by_default(visibility))


def _cc_proto_library_impl(_package: Package,
                           _label: Label,
                           deps: Optional[List[RelativeLabel]] = None,
                           **kwargs):
  del kwargs
  context = _package.context
  cmake_target_pair = context.generate_cmake_target_pair(
      _label, generate_alias=True)
  dep_targets = _package.get_label_list(deps)
  dep_library_targets = [
      _get_cc_proto_library_target(_package.context, dep_target)
      for dep_target in dep_targets
  ]
  _emit_cc_library(
      context.builder,
      cmake_target_pair,
      hdrs=set(),
      srcs=set(),
      deps=set(context.get_deps(dep_library_targets)),
  )
  context.add_analyzed_target(_label,
                              TargetInfo(*cmake_target_pair.as_providers()))


def _get_cc_proto_library_target(context: EvaluationContext,
                                 proto_target: Label) -> Label:
  proto_info = context.get_target_info(proto_target)[ProtoLibraryProvider]
  cc_deps = [
      _get_single_cc_proto_target(context, proto_src, proto_info.deps)
      for proto_src in proto_info.srcs
  ]
  if len(cc_deps) == 1:
    return cc_deps[0]

  cc_library_target = proto_target + "__cc_proto_library"
  cmake_target_pair = context.generate_cmake_target_pair(
      cc_library_target, generate_alias=True)

  _emit_cc_library(
      context.builder,
      cmake_target_pair,
      hdrs=set(),
      srcs=set(),
      deps=set(context.get_deps(cc_deps)),
  )
  return cc_library_target


def _get_single_cc_proto_target(context: EvaluationContext, proto_src: Label,
                                deps: Set[Label]) -> Label:
  protoc_output_target = proto_src + "__cc_protoc"
  cc_library_target = proto_src + "__cc_proto"

  info = context.get_optional_target_info(cc_library_target)
  if info is not None:
    return cc_library_target

  proto_suffix = ".proto"
  assert proto_src.endswith(proto_suffix)
  proto_prefix = proto_src[:-len(proto_suffix)]
  generated_pb_h = f"{proto_prefix}.pb.h"
  generated_pb_cc = f"{proto_prefix}.pb.cc"

  cmake_protoc_target_pair = context.generate_cmake_target_pair(
      protoc_output_target, generate_alias=False)
  cmake_protoc_deps = CMakeDepsProvider([cmake_protoc_target_pair.target])

  cmake_cc_target_pair = context.generate_cmake_target_pair(
      cc_library_target, generate_alias=True)

  generated_pb_h_path = context.get_generated_file_path(generated_pb_h)
  generated_pb_cc_path = context.get_generated_file_path(generated_pb_cc)

  context.add_analyzed_target(
      generated_pb_h,
      TargetInfo(FilesProvider([generated_pb_h_path]), cmake_protoc_deps))
  context.add_analyzed_target(
      generated_pb_cc,
      TargetInfo(FilesProvider([generated_pb_cc_path]), cmake_protoc_deps))

  context.add_analyzed_target(protoc_output_target,
                              TargetInfo(cmake_protoc_deps))

  context.add_analyzed_target(cc_library_target,
                              TargetInfo(*cmake_cc_target_pair.as_providers()))

  cmake_deps: List[CMakeTarget] = []
  proto_src_files = context.get_file_paths(proto_src, cmake_deps)
  assert len(proto_src_files) == 1
  proto_src_file = proto_src_files[0]
  _emit_cc_proto_generate(
      context.builder,
      cmake_protoc_target_pair.target,
      proto_src=proto_src_file,
      cmake_deps=cmake_deps,
      generated_pb_h=generated_pb_h_path,
      generated_pb_cc=generated_pb_cc_path)

  cc_deps = [
      _get_cc_proto_library_target(context, proto_dep) for proto_dep in deps
  ]
  cc_deps.append("@com_google_protobuf//:protobuf")

  cmake_cc_deps = context.get_deps(cc_deps)

  _emit_cc_library(
      context.builder,
      cmake_cc_target_pair,
      hdrs=set([generated_pb_h_path]),
      srcs=set([generated_pb_cc_path]),
      deps=set(cmake_cc_deps),
  )
  context.builder.addtext(
      f"add_dependencies({cmake_cc_target_pair.target} {cmake_protoc_target_pair.target})\n"
  )
  return cc_library_target


def _emit_cc_proto_generate(
    _builder: CMakeBuilder,
    cmake_target: str,
    proto_src: str,
    generated_pb_h: str,
    generated_pb_cc: str,
    cmake_deps: List[str],
):
  """Generates a C++ library corresponding to a Protobuf."""
  cmake_deps.append("protobuf::protoc")
  cmake_deps.append(proto_src)
  _builder.addtext(f"""
add_custom_command(
  OUTPUT {quote_list([generated_pb_h, generated_pb_cc])}
  COMMAND protobuf::protoc
  ARGS --experimental_allow_proto3_optional --cpp_out "${{PROJECT_BINARY_DIR}}" -I "${{PROJECT_SOURCE_DIR}}" {cmake_builder.quote_path(proto_src)}
  DEPENDS {quote_list(cmake_deps)}
  COMMENT "Running cpp protocol buffer compiler on {proto_src}"
  VERBATIM)
add_custom_target({cmake_target} DEPENDS {quote_list([generated_pb_h, generated_pb_cc])})
""")


@register_native_build_rule
def genrule(self: EvaluationContext,
            name: str,
            outs: List[RelativeLabel],
            visibility: Optional[List[RelativeLabel]] = None,
            **kwargs):
  package = self.current_package
  assert package is not None
  label = package.get_label(name)
  out_targets = [package.get_label(out) for out in outs]

  self.add_rule(
      label,
      lambda: _genrule_impl(
          cast(Package, package), label, out_targets, **kwargs),
      outs=out_targets,
      analyze_by_default=package.analyze_by_default(visibility))


def _get_relative_path(path: str, relative_to: str) -> str:
  rel_path = os.path.relpath(path, relative_to)
  if os.sep != "/":
    rel_path = rel_path.replace(os.sep, "/")
  return rel_path


def _genrule_impl(_package: Package,
                  _label: Label,
                  _out_targets: List[Label],
                  cmd: Configurable[str],
                  srcs: Optional[Configurable[List[RelativeLabel]]] = None,
                  message: Optional[Configurable[str]] = None,
                  toolchains: Optional[List[RelativeLabel]] = None,
                  **kwargs):
  context = _package.context
  cmake_target_pair = context.generate_cmake_target_pair(
      _label, generate_alias=False)
  cmake_deps_provider = CMakeDepsProvider([cmake_target_pair.target])
  out_files: List[str] = []
  for out_target in _out_targets:
    out_file = context.get_generated_file_path(out_target)
    out_files.append(out_file)
    context.add_analyzed_target(
        out_target, TargetInfo(FilesProvider([out_file]), cmake_deps_provider))
  del kwargs
  message_text = ""
  if message is not None:
    message_text = _package.context.evaluate_configurable(message)
  cmake_deps: List[CMakeTarget] = []
  src_files = _package.context.get_targets_file_paths(
      _package.get_label_list(srcs), cmake_deps)
  cmake_deps.extend(src_files)
  relative_source_paths = [
      _get_relative_path(path, _package.repo.source_directory)
      for path in src_files
  ]
  relative_out_paths = [
      _get_relative_path(path, _package.repo.source_directory)
      for path in out_files
  ]
  substitutions = {
      "$(SRCS)": " ".join(relative_source_paths),
      "$(OUTS)": " ".join(relative_out_paths),
  }
  if len(out_files) == 1:
    substitutions["$@"] = relative_out_paths[0]
  if len(src_files) == 1:
    substitutions["$<"] = relative_source_paths[0]
  # TODO(jbms): Add missing variables, including:
  #   "$(BINDIR)"
  #   "$(GENDIR)"
  #   "$(COMPILATION_MODE)"
  #   "$(TARGET_CPU)"
  #   "$(RULEDIR)"
  #   "$(@D)"
  cmd_text = _package.context.evaluate_configurable(cmd)
  cmd_text = apply_location_substitutions(
      _package, cmd_text, relative_to=_package.repo.source_directory)
  cmd_text = apply_make_variable_substitutions(
      _package.context.builder,
      cmd_text,
      substitutions=substitutions,
      toolchains=_package.get_label_list(toolchains))
  _emit_genrule(
      builder=_package.context.builder,
      cmake_target=cmake_deps_provider.targets[0],
      out_files=out_files,
      cmake_deps=cmake_deps,
      cmd_text=cmd_text,
      message=message_text)
  context.add_analyzed_target(_label, TargetInfo(cmake_deps_provider))


def _emit_genrule(
    builder: CMakeBuilder,
    cmake_target: str,
    out_files: List[str],
    cmake_deps: List[str],
    cmd_text: str,
    message: Optional[str] = None,
):
  if message:
    optional_message_text = f"COMMENT {cmake_builder.quote_string(message)}"
  else:
    optional_message_text = ""
  builder.addtext(f"""
add_custom_command(
  OUTPUT {cmake_builder.quote_list(out_files)}
  DEPENDS {cmake_builder.quote_list(cmake_deps)}
  COMMAND {cmd_text}
  {optional_message_text}
  VERBATIM
  WORKING_DIRECTORY "${{CMAKE_CURRENT_SOURCE_DIR}}"
)
add_custom_target({cmake_target} DEPENDS {cmake_builder.quote_list(out_files)})
""")


@register_native_build_rule
def config_setting(
    self: EvaluationContext,
    name: str,
    constraint_values: Optional[List[RelativeLabel]] = None,
    flag_values: Optional[Dict[RelativeLabel, str]] = None,
    values: Optional[Dict[str, str]] = None,
    define_values: Optional[Dict[str, str]] = None,
    visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  del kwargs
  # Bazel ignores visibility for `config_setting` by default.  See
  # `--incompatible_enforce_config_setting_visibility` and
  # `--incompatible_config_setting_private_default_visibility`.
  del visibility
  package = self.current_package
  assert package is not None
  if constraint_values:
    resolved_constraint_values = [
        package.get_label(condition) for condition in constraint_values
    ]
  if flag_values:
    resolved_flag_values = {
        package.get_label(flag): value for flag, value in flag_values.items()
    }

  def evaluate() -> bool:
    if flag_values:
      for flag, value in resolved_flag_values.items():
        if self.evaluate_build_setting(flag) != value:
          return False
    if constraint_values:
      for constraint in resolved_constraint_values:
        if not self.evaluate_condition(constraint):
          return False
    if values:
      for key, value in values.items():
        if (key, value) not in self.workspace.values:
          return False
    if define_values:
      for key, value in define_values.items():
        if ("define", f"{key}={value}") not in self.workspace.values:
          return False
    return True

  label = package.get_label(name)

  self.add_rule(
      label,
      lambda: self.add_analyzed_target(
          label, TargetInfo(ConditionProvider(evaluate()))),
      analyze_by_default=True)


@register_native_build_rule
def alias(self: EvaluationContext,
          name: str,
          actual: Configurable[RelativeLabel],
          visibility: Optional[List[RelativeLabel]] = None,
          **kwargs):
  del kwargs
  package = self.current_package
  assert package is not None
  label = package.get_label(name)

  def impl():
    self.add_analyzed_target(
        label,
        self.get_target_info(
            package.get_label(self.evaluate_configurable(actual))))

  self.add_rule(
      label, impl, analyze_by_default=package.analyze_by_default(visibility))


@register_native_build_rule
def exports_files(self: EvaluationContext, *args, **kwargs):
  del self
  del args
  del kwargs


@register_native_build_rule
def py_library(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def py_test(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def py_binary(self: EvaluationContext, name: str, **kwargs):
  del self
  del name
  del kwargs
