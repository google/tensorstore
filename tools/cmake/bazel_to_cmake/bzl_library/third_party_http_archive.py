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
"""CMake implementation of "@com_google_tensorstore//third_party:repo.bzl".

The following parameters of `third_party_http_archive` are supported:

- name: str

  Specifies the Bazel repo name.  Used by both Bazel and CMake.

- repo_mapping: Optional[Dict[str, str]]

  Optional.  Maps repository names used by the third-party package to the name
  the repository is available in the top-level workspace.  Used by Bazel, and by
  CMake if `bazel_to_cmake` is specified.

- urls: List[str]

  Required.  Specifies the list of URLs.  Used by both Bazel and CMake.

- sha256: str

  Optional (but should always be specified).  Specifies the sha256 hash in hex
  representation of the archive specified by `urls`.  Used by both Bazel and
  CMake.

- strip_prefix: Optional[str]

  Optional.  Top-level directory to strip when extracting the archive.  Used
  only by Bazel.  CMake seems to auto-detect.

- patches: List[Label]

  Optional.  List of patches to apply.  Used by both Bazel and CMake.

- patch_args: Optional[List[str]]

  Optional.  List of arguments to pass to `patch`, e.g. `["-p1"]` for patches
  produced by `git format-patch`.  Used by both Bazel and CMake.

- patch_cmds: Optional[List[str]]

  Optional.  Specifies list of additional patch commands to perform after
  applying the patches specified by `patches`.  Used by both Bazel and CMake.

- remove_paths: Optional[List[str]]

  Optional.  Specifies a list of files or directories to delete as an additional
  patch action.  Used by both Bazel and CMake.

- build_file: Optional[Label]

  Optional.  Specifies Bazel build file to use (when not using a system version
  of the library).  Overrides any Bazel build file already present.  Used by
  Bazel, and by CMake when `bazel_to_cmake` is specified.

- system_build_file: Optional[Label]

  Optional.  Specifies Bazel build file to use when using a system version of
  the library.  Used only by Bazel.  On CMake, a `FindXXX.cmake` module must be
  provided instead.

- doc_name: Optional[str]

  Optional. Specifies the package name for display in the documentation.  Not
  used when building.

- doc_version: Optional[str]

  Optional.  Specifies the version number for display in the documentation.  Not
  used when building.

- doc_homepage: Optional[str]

  Optional.  Specifies the homepage URL for display in the documentation.  Not
  used when building.

- cmake_name: Optional[str]

  Optional.  Specifies the CMake project name.  Should match the identifier
  typically usde with `find_package`.  Used only by CMake.  If not specified,
  the repository will be ignored in the CMake build.

- bazel_to_cmake: Optional[Dict[str, Any]]

  Optional.  Used only by CMake.  If specified, `bazel_to_cmake` is used on the
  dependency in place of a "native" CMake build.  Supported members are:

  - include: Optional[List[str]]

    List of glob patterns matching package names to include in the build,
    e.g. `["", "tensorstore/**"]` to include the top-level package as well as
    all packages under "tensorstore".

  - exclude: Optional[List[str]]

    List of glob patterns matching package names to exclude from the build.

  - aliased_targets_only: bool

    Optional.  If set to `True`, only analyze (and generated CMake targets for)
    the transitive dependencies of targets listed in `cmake_target_mapping`.  If
    `False` (the default), all targets (except `filegroup`) are analyzed.

- cmake_target_mapping: Optional[Dict[RelativeLabel, CMakeTarget]]

  Optional.  Used only by CMake.  Maps Bazel targets to corresponding CMake
  target names.  Bazel targets may be relative, in which case they are resolved
  relative to the repository `name`.  The CMake target name should be the name
  assigned by existing `FindXXX.cmake` modules, if they exist.  Typically, if
  there is just a single library target, it is named `NAME::NAME`,
  e.g. `JPEG::JPEG`, where `NAME` is the `cmake_name`.  If `bazel_to_cmake` is
  specified, these mappings will be generated.  Otherwise, the specified CMake
  targets must be defined by the existing CMake build (or by
  `cmakelists_suffix` or `cmake_aliases`).

- cmakelists_prefix: Optional[str]

  Optional.  Used only by CMake.  Additional content to prepend to the CMake
  build configuration for this repository (applies even if `bazel_to_cmake` is
  specified).

- cmakelists_suffix: Optional[str]

  Optional.  Used only by CMake.  Additional content to append to the CMake
  build configuration for this repository (applies even if `bazel_to_cmake` is
  specified).  This may be used to add additional target properties or define
  additional aliases.

- cmake_settings: Optional[Dict[str, str]]

  Optional.  Used only by CMake.  CMake variables to set when configuring this
  repository.  These are not added to the CMake cache.  Note that due to the way
  `FetchContent` works, they may also apply to some dependencies of this
  repository.

- cmake_languages: Optional[List[str]]

  Optional.  Used only by CMake.  List of languages for which `enable_language`
  must be called.  These must be specified since CMake requires all
  `enable_language` calls to happen at the top-level.

- cmake_source_subdir: Optional[str]

  Optional.  Used only by CMake.  Relative path to the directory containing the
  top-level `CMakeLists.txt` file.

- cmake_package_redirect_extra: Optional[str]

  Optional.  Used only by CMake.  Additional content to execute when
  `find_package(<NAME>)` is called, where `<NAME>` is equal to `cmake_name`.

  CMake defines `<NAME>_FOUND`, but in order for other CMake projects to
  correctly used this repository as a dependency, other CMake variables may be
  necessary.

  If additional target aliases are needed, they should be defined using
  `cmake_aliases` or `cmakelists_suffix` instead.

  In most cases, the `cmake_package_redirect_libraries` option provides a more
  convenient way to add content to execute.

- cmake_package_redirect_libraries: Optional[Dict[str, CMakeTarget]]

  Optional.  Used only by CMake.  Maps cmake variable names prefixes to
  corresponding CMake targets.

  For each entry `(var, target)`, the following additional variables are defined
  when `find_package(<NAME>)` is called:

      set(var_LIBRARY target)
      set(var_LIBRARIES target)
      set(VAR_LIBRARY target)
      set(VAR_LIBRARIES target)
      set(var_INCLUDE_DIR <INTERFACE_INCLUDE_DIRECTORIES of target>)
      set(var_INCLUDE_DIRS <INTERFACE_INCLUDE_DIRECTORIES of target>)
      set(VAR_INCLUDE_DIR <INTERFACE_INCLUDE_DIRECTORIES of target>)
      set(VAR_INCLUDE_DIRS <INTERFACE_INCLUDE_DIRECTORIES of target>)

  Modern CMake builds should just use the target, but some projects rely on
  these variables.

- cmake_package_aliases: Optional[List[str]]

  Optional.  Used only by CMake.  List of additional names for which
  `find_package(<NAME>)` will find this package and execute the content
  specified by `cmake_package_redirect_extra` and
  `cmake_package_redirect_libraries`.

"""

# pylint: disable=invalid-name,missing-function-docstring,relative-beyond-top-level,g-long-lambda

import io
import os
import pathlib
import sys
from typing import Dict, List, Optional, Any

from .. import cmake_builder
from ..evaluation import BazelGlobals
from ..evaluation import EvaluationContext
from ..evaluation import register_bzl_library
from ..label import CMakeTarget
from ..label import Label
from ..label import label_to_generated_cmake_target
from ..label import resolve_label
from ..util import cmake_is_true
from ..workspace import Repository


@register_bzl_library(
    "@com_google_tensorstore//third_party:repo.bzl", workspace=True)
class ThirdPartyRepoLibrary(BazelGlobals):

  def bazel_third_party_http_archive(self, **kwargs):
    _third_party_http_archive_impl(self._context, **kwargs)

  def bazel_third_party_python_package(self, *args, **kwargs):
    del args
    del kwargs
    pass


def _get_fetch_content_third_party_dir(context: EvaluationContext) -> str:
  return os.path.join(context.repo.cmake_binary_dir, "third_party")


def _get_file_path(_context: EvaluationContext, _repo: Repository,
                   label: Label) -> str:
  path = _context.get_source_file_path(_repo.get_label(label))
  assert path is not None
  return path


def _get_fetch_content_invocation(
    _repo: Repository,
    _context: EvaluationContext,
    _builder: cmake_builder.CMakeBuilder,
    name: str,
    cmake_name: str,
    _cmake_reverse_target_mapping: Dict[CMakeTarget, Label],
    urls: Optional[List[str]] = None,
    sha256: Optional[str] = None,
    patch_args: Optional[List[str]] = None,
    patches: Optional[List[Label]] = None,
    patch_cmds: Optional[List[str]] = None,
    remove_paths: Optional[List[str]] = None,
    **kwargs,
) -> str:
  """Convert `third_party_http_archive` options to CMake FetchContent invocation."""
  cmake_command = _context.workspace.cmake_vars["CMAKE_COMMAND"]
  out = io.StringIO()
  out.write(f"FetchContent_Declare({cmake_name}")
  if urls:
    out.write(f" URL {cmake_builder.quote_string(urls[0])}")
  if sha256:
    hash_str = f"SHA256={sha256}"
    out.write(f" URL_HASH {cmake_builder.quote_string(hash_str)}")

  patch_commands = []
  for patch in patches or ():
    quoted_patch_path = cmake_builder.quote_path(
        _get_file_path(_context, _repo, patch))
    patch_commands.append(
        f"""${{Patch_EXECUTABLE}} --binary {" ".join(patch_args or ())} < {quoted_patch_path}"""
    )
  if patches:
    _builder.find_package("Patch")
  if patch_cmds:
    patch_commands.extend(patch_cmds)
  if remove_paths:
    remove_arg = " ".join(
        cmake_builder.quote_path(path) for path in remove_paths)
    patch_commands.append(
        f"{cmake_builder.quote_path(cmake_command)} -E rm -rf {remove_arg}")
  new_cmakelists_path = os.path.join(
      _get_fetch_content_third_party_dir(_context),
      f"{cmake_name}-proxy-CMakeLists.txt")
  pathlib.Path(new_cmakelists_path).write_text(
      _get_subproject_cmakelists(
          _context=_context,
          _repo=_repo,
          _patch_commands=patch_commands,
          name=name,
          cmake_name=cmake_name,
          **kwargs),
      encoding="utf-8")
  patch_commands.append(
      f"""{cmake_builder.quote_path(cmake_command)} -E copy {cmake_builder.quote_path(new_cmakelists_path)} CMakeLists.txt"""
  )
  patch_command = " && ".join(patch_commands)
  out.write(f" PATCH_COMMAND {patch_command}")
  out.write(" OVERRIDE_FIND_PACKAGE)\n")
  return out.getvalue()


def _get_subproject_cmakelists(_context: EvaluationContext,
                               _repo: Repository,
                               _patch_commands: List[str],
                               name: str,
                               cmake_name: str,
                               cmakelists_prefix: Optional[str] = None,
                               cmakelists_suffix: Optional[str] = None,
                               bazel_to_cmake: Optional[Dict[str, Any]] = None,
                               cmake_target_mapping: Optional[Dict[
                                   Label, CMakeTarget]] = None,
                               cmake_source_subdir: Optional[str] = None,
                               cmake_settings: Optional[Dict[str, str]] = None,
                               cmake_aliases: Optional[Dict[str, str]] = None,
                               **kwargs) -> str:
  new_cmakelists = io.StringIO()
  cmake_command = _context.workspace.cmake_vars["CMAKE_COMMAND"]
  new_cmakelists.write(f'set(CMAKE_MESSAGE_INDENT "[{cmake_name}] ")\n')

  # Workaround for https://gitlab.kitware.com/cmake/cmake/-/issues/24013
  #
  # Due to the way `FetchContent` works, if library `A` depends on library `B`
  # (and therefore library `A` calls `find_package(B)`), and library `A`
  # happens to be listed before library `B` in the call to
  # `FetchContent_MakeAvailable`, then library `B` will actually be added as a
  # sub-directory of library `A` and inherit any directory properties of
  # library `A` that were set prior to the call to `find_package`, which is
  # undesirable.
  #
  # In particular, that can lead to compiler flags set by library A also
  # applying to library B.
  #
  # To avoid that, we explicitly override the value of important directory
  # properties.
  for prop in [
      "COMPILE_DEFINITIONS",
      "COMPILE_OPTIONS",
      "INCLUDE_DIRECTORIES",
      "LINK_DIRECTORIES",
      "LINK_OPTIONS",
  ]:
    new_cmakelists.write(f"""
get_property(_prop DIRECTORY {cmake_builder.quote_path(_repo.source_directory)} PROPERTY {prop})
set_property(DIRECTORY PROPERTY {prop} "${{_prop}}")
unset(_prop)
""")

  # These variable are set by `FetchContent`, and with CMP0126 enabled, won't
  # be overridden by the `project` command, which causes problems for packages
  # where the `project` with the same name as the package is actually defined
  # in a sub-directory.
  new_cmakelists.write(f"unset({cmake_name.lower()}_BINARY_DIR)\n")
  new_cmakelists.write(f"unset({cmake_name.lower()}_SOURCE_DIR)\n")

  if cmakelists_prefix:
    new_cmakelists.write(cmakelists_prefix)

  if bazel_to_cmake is not None:
    _write_bazel_to_cmake_cmakelists(
        _new_cmakelists=new_cmakelists,
        _patch_commands=_patch_commands,
        _context=_context,
        _repo=_repo,
        name=name,
        cmake_name=cmake_name,
        bazel_to_cmake=bazel_to_cmake,
        cmake_target_mapping=cmake_target_mapping,
        **kwargs)
  else:
    if not cmake_source_subdir:
      _patch_commands.append(
          f"{cmake_builder.quote_path(cmake_command)} -E copy CMakeLists.txt orig_CMakeLists.cmake"
      )
    for k, v in (cmake_settings or {}).items():
      new_cmakelists.write(f"""set({k} "{v}")\n""")
    if cmake_source_subdir:
      new_cmakelists.write(
          f"""add_subdirectory({cmake_builder.quote_path(cmake_source_subdir)})\n"""
      )
    else:
      new_cmakelists.write(
          """include("${CMAKE_CURRENT_LIST_DIR}/orig_CMakeLists.cmake")\n""")
  for alias_target, orig_target in (cmake_aliases or {}).items():
    new_cmakelists.write(f"""
get_property(_aliased TARGET {orig_target} PROPERTY ALIASED_TARGET)
get_property(_type TARGET {orig_target} PROPERTY TYPE)
if("${{_aliased}}" STREQUAL "")
  set(_aliased {cmake_builder.quote_string(orig_target)})
endif()
if("${{_type}}" STREQUAL "EXECUTABLE")
  add_executable({alias_target} ALIAS "${{_aliased}}")
else()
  add_library({alias_target} ALIAS "${{_aliased}}")
endif()
""")
  if cmakelists_suffix:
    new_cmakelists.write(cmakelists_suffix)

  return new_cmakelists.getvalue()


def _write_bazel_to_cmake_cmakelists(
    _new_cmakelists: io.StringIO,
    _patch_commands: List[str],
    _context: EvaluationContext,
    _repo: Repository,
    name: str,
    cmake_name: str,
    bazel_to_cmake: Dict[str, Any],
    cmake_target_mapping: Optional[Dict[Label, CMakeTarget]] = None,
    build_file: Optional[Label] = None,
    repo_mapping: Optional[Dict[str, str]] = None,
    **kwargs):
  del kwargs
  if build_file is not None:
    cmake_command = _context.workspace.cmake_vars["CMAKE_COMMAND"]
    quoted_build_path = cmake_builder.quote_path(
        _get_file_path(_context, _repo, build_file))
    _patch_commands.append(
        f"""{cmake_builder.quote_path(cmake_command)} -E copy {quoted_build_path} BUILD.bazel"""
    )
  bazel_to_cmake_path = os.path.abspath(sys.argv[0])
  bazel_to_cmake_cmd = f"${{Python3_EXECUTABLE}} {cmake_builder.quote_path(bazel_to_cmake_path)}"
  assert _context.save_workspace is not None
  bazel_to_cmake_cmd += (
      f" --load-workspace {cmake_builder.quote_path(_context.save_workspace)}")
  bazel_to_cmake_cmd += f" --cmake-project-name {cmake_name}"
  bazel_to_cmake_cmd += ' --cmake-binary-dir "${CMAKE_CURRENT_BINARY_DIR}"'
  bazel_to_cmake_cmd += f" --bazel-repo-name {name}"
  bazel_to_cmake_cmd += (
      ' --build-rules-output "${CMAKE_CURRENT_BINARY_DIR}/build_rules.cmake"')
  for mapped, orig in (repo_mapping or {}).items():
    bazel_to_cmake_cmd += f" --repo-mapping {mapped} {orig}"
  for include_package in bazel_to_cmake.get("include", []):
    bazel_to_cmake_cmd += " " + cmake_builder.quote_string(
        "--include-package=" + include_package)
  for exclude_package in bazel_to_cmake.get("exclude", []):
    bazel_to_cmake_cmd += " " + cmake_builder.quote_string(
        "--exclude-package=" + exclude_package)
  if bazel_to_cmake.get("aliased_targets_only"):
    for target in (cmake_target_mapping or {}):
      bazel_to_cmake_cmd += f" --target {cmake_builder.quote_string(target)}"
  for bazel_target, cmake_alias in (cmake_target_mapping or {}).items():
    bazel_to_cmake_cmd += f" --target-alias {cmake_builder.quote_string(bazel_target)} {cmake_builder.quote_string(cmake_alias)}"

  bazel_to_cmake_cmd += " " + " ".join(bazel_to_cmake.get("args", []))
  _new_cmakelists.write(f"""
project({cmake_builder.quote_string(cmake_name)})
execute_process(
  COMMAND {bazel_to_cmake_cmd}
  WORKING_DIRECTORY "${{CMAKE_CURRENT_SOURCE_DIR}}"
  COMMAND_ERROR_IS_FATAL ANY)
include("${{CMAKE_CURRENT_BINARY_DIR}}/build_rules.cmake")
""")


_FETCH_CONTENT_PACKAGES_KEY = "fetch_content_packages"


def _third_party_http_archive_impl(_context: EvaluationContext, **kwargs):
  builder = _context.builder
  bazel_name = kwargs["name"]
  cmake_name = kwargs.get("cmake_name")
  repo = _context.repo
  if not cmake_name:
    return

  if not kwargs.get("urls"):
    return

  bazel_to_cmake = kwargs.get("bazel_to_cmake")
  if bazel_to_cmake is not None:
    repo.workspace.bazel_to_cmake_deps[bazel_name] = cmake_name

  new_base_package = f"@{bazel_name}//"

  target_mapping = kwargs.get("cmake_target_mapping")
  canonical_target_mapping: Dict[Label, CMakeTarget] = {}
  reverse_target_mapping: Dict[CMakeTarget, Label] = {}
  if target_mapping:
    for relative_label, cmake_target in target_mapping.items():
      label = resolve_label(relative_label, base_package=new_base_package)
      reverse_target_mapping.setdefault(cmake_target, label)
      repo.workspace.set_bazel_target_mapping(
          label, cmake_target, cmake_package=cmake_name)
      canonical_target_mapping[label] = cmake_target
  kwargs["cmake_target_mapping"] = canonical_target_mapping
  kwargs["_cmake_reverse_target_mapping"] = reverse_target_mapping

  # TODO(jbms): Use some criteria (e.g. presence of system_build_file option) to
  # determine whether to support a system library, rather than always using it.
  use_system_option = f"TENSORSTORE_USE_SYSTEM_{cmake_name.upper()}"
  builder.addtext(
      f"""option({use_system_option} "Use an installed version of {cmake_name}")\n""",
      section=cmake_builder.OPTIONS_SECTION,
  )
  if cmake_is_true(_context.workspace.cmake_vars.get(use_system_option)):
    return

  builder.include("FetchContent")
  for lang in kwargs.pop("cmake_languages", []):
    builder.addtext(
        f"enable_language({lang})\n",
        section=cmake_builder.FETCH_CONTENT_DECLARE_SECTION,
        unique=True,
    )

  fetch_content_packages = getattr(_context, _FETCH_CONTENT_PACKAGES_KEY, None)
  if fetch_content_packages is None:
    # Create the "${PROJECT_BINARY_DIR}/third_party" directory used for writing
    # the `CMakeLists.txt` files for each dependency.  Since the dependency has
    # not yet been populated, we can't yet write the `CMakeLists.txt` file to
    # its final location.  Instead, write it to this `third_party` directory,
    # and add a patch command to copy it over when the dependency is populated.
    third_party_dir = _get_fetch_content_third_party_dir(_context)
    os.makedirs(third_party_dir, exist_ok=True)

    fetch_content_packages = []
    setattr(_context, _FETCH_CONTENT_PACKAGES_KEY, fetch_content_packages)
    # At the end of the analysis phase, emit additional code to call
    # `FetchContent_MakeAvailable`.
    _context.call_after_analysis(
        lambda: _emit_fetch_content_make_available(_context))
  fetch_content_packages.append(cmake_name)

  builder.addtext(
      _get_fetch_content_invocation(
          _repo=_context.repo, _context=_context, _builder=builder, **kwargs),
      section=cmake_builder.FETCH_CONTENT_DECLARE_SECTION,
  )

  extra = kwargs.get("cmake_package_redirect_extra", "")
  package_aliases = kwargs.get("cmake_package_aliases", [])
  if cmake_name != cmake_name.upper() and cmake_name.upper():
    package_aliases.append(cmake_name.upper())
  package_aliases = sorted(set(package_aliases))
  cmake_find_package_redirects_dir = _context.workspace.cmake_vars[
      "CMAKE_FIND_PACKAGE_REDIRECTS_DIR"]
  for alias in package_aliases:
    if alias.lower() == cmake_name.lower():
      continue
    config_path = os.path.join(cmake_find_package_redirects_dir,
                               f"{alias.lower()}-config.cmake")
    config_path_alt = os.path.join(cmake_find_package_redirects_dir,
                                   f"{alias}Config.cmake")
    if not os.path.exists(config_path) and not os.path.exists(config_path_alt):
      pathlib.Path(config_path).write_text(
          f"""
include(CMakeFindDependencyMacro)
find_dependency({cmake_name})
""",
          encoding="utf-8")

  cmake_package_redirect_libraries = kwargs.get(
      "cmake_package_redirect_libraries", {})
  if extra or package_aliases or cmake_package_redirect_libraries:
    extra_aliases = ""
    for alias in package_aliases:
      extra_aliases += f"set({alias}_FOUND ON)\n"
    for var_prefix, cmake_target in cmake_package_redirect_libraries.items():
      # When using bazel_to_cmake, map `target` to the non-aliased target.
      t = reverse_target_mapping.get(cmake_target)
      if t is not None:
        cmake_target = label_to_generated_cmake_target(
            bazel_target=t, cmake_project=cmake_name)
      for suffix in ("LIBRARY", "LIBRARIES"):
        extra_aliases += f"set({var_prefix}_{suffix} {cmake_target})\n"
        if var_prefix != var_prefix.upper():
          extra_aliases += f"set({var_prefix.upper()}_{suffix} {cmake_target})\n"
      for suffix in ("INCLUDE_DIR", "INCLUDE_DIRS"):
        extra_aliases += f"get_property({var_prefix}_{suffix} TARGET {cmake_target} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)\n"
        if var_prefix != var_prefix.upper():
          extra_aliases += f"get_property({var_prefix.upper()}_{suffix} TARGET {cmake_target} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)\n"
    extra_path = os.path.join(cmake_find_package_redirects_dir,
                              f"{cmake_name.lower()}-extra.cmake")
    extra_path_alt = os.path.join(cmake_find_package_redirects_dir,
                                  f"{cmake_name}Extra.cmake")
    if not os.path.exists(extra_path) and not os.path.exists(extra_path_alt):
      pathlib.Path(extra_path).write_text(
          extra + "\n" + extra_aliases, encoding="utf-8")


def _emit_fetch_content_make_available(context: EvaluationContext):
  """Emit CMake code to call `FetchContent_MakeAvailable`.

  If `FetchContent_MakeAvailable` is invoked directly from main
  `build_rules.cmake` file, there is no option to specify `EXCLUDE_FROM_ALL`.

  As a workaround, write the `FetchContent_MakeAvailable` call to a
  `CMakeLists.txt` file in a sub-directory, and then load that sub-directory
  with `EXCLUDE_FROM_ALL`.  Since `EXCLUDE_FROM_ALL` is inherited, that ensures
  all dependencies are loaded with `EXCLUDE_FROM_ALL`.

  Args:
    context: Evaluation context.
  """
  third_party_dir = _get_fetch_content_third_party_dir(context)
  fetch_content_packages: List[str] = getattr(context,
                                              _FETCH_CONTENT_PACKAGES_KEY)
  # When a subdirectory is added to CMake, it has both a "source directory" and
  # a "binary directory", corresponding to `CMAKE_CURRENT_SOURCE_DIR` and
  # `CMAKE_CURRENT_BINARY_DIR`, respectively, when evaluating the subdirectory
  # CMakeLists.txt file.  If the "binary directory" is unspecified for a
  # sub-directory, CMake will normally choose a binary directory automatically,
  # using the relative path of the source directory from the current source
  # directory.  However, in this case the "source directory" (`third_party_dir`)
  # is a sub-directory of the current binary dir, which is not in general a
  # descendant of the project source directory, and therefore the binary
  # directory ("_third_party_configs") must be specified explicitly.  Note that
  # this binary directory is not actually used for much, since no targets are
  # defined in this sub-directory.
  context.builder.addtext(
      f"add_subdirectory({cmake_builder.quote_path(third_party_dir)} _third_party_configs EXCLUDE_FROM_ALL)\n",
      section=cmake_builder.FETCH_CONTENT_MAKE_AVAILABLE_SECTION,
  )
  make_available_args = cmake_builder.quote_list(
      sorted(pkg for pkg in fetch_content_packages
             if pkg in context.required_dep_packages))
  third_party_cmakelists_path = os.path.join(third_party_dir, "CMakeLists.txt")
  pathlib.Path(third_party_cmakelists_path).write_text(
      f"""
include(FetchContent)
FetchContent_MakeAvailable({make_available_args})
""",
      encoding="utf-8")
