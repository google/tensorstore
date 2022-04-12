#!/usr/bin/env python3
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
"""Builds CMakeLists.txt files from bazel BUILD files.

The basic idea is that the BUILD, WORKSPACE, and workspace.bzl files include
configuration using 'dummy' bazel macros that are then interpreted by this
script to generate CMakeLists.txt files in each subdirectory.

To use:

  cd /path/to/tensorstore
  python3 CMake/bazel_to_cmake.py

This is very much still a work in progress.
"""

from __future__ import print_function

import collections
import glob
import os
import re
import sys
import traceback
import types
from typing import Any, Dict, List, Optional, Set, Tuple

# Only generate CMakeLists.txt for targets up to this depth, by default.
# If the KEY_DEPTH is set to 1 (for tensorstore), the resulting file
# (tensorstore/CMakeLists.txt) is around 12k lines long.
KEY_DEPTH = 2

FIND_PACKAGE_OPTIONS = [
    "EXACT", "QUIET", "REQUIRED", "CONFIG", "NO_MODULE", "NO_POLICY_SCOPE",
    "COMPONENTS", "OPTIONAL_COMPONENTS", "NAMES", "CONFIGS", "HINTS", "PATHS",
    "PATH_SUFFIXES", "NO_DEFAULT_PATH", "NO_PACKAGE_ROOT_PATH", "NO_CMAKE_PATH",
    "NO_CMAKE_ENVIRONMENT_PATH", "NO_SYSTEM_ENVIRONMENT_PATH",
    "NO_CMAKE_PACKAGE_REGISTRY", "NO_CMAKE_SYSTEM_PATH",
    "NO_CMAKE_SYSTEM_PACKAGE_REGISTRY", "CMAKE_FIND_ROOT_PATH_BOTH",
    "ONLY_CMAKE_FIND_ROOT_PATH", "NO_CMAKE_FIND_ROOT_PATH"
]

FETCH_CONTENT_DECLARE_OPTIONS = [
    "PREFIX", "TMP_DIR", "STAMP_DIR", "LOG_DIR", "DOWNLOAD_DIR", "SOURCE_DIR",
    "BINARY_DIR", "INSTALL_DIR", "DOWNLOAD_COMMAND", "URL", "URL_HASH",
    "DOWNLOAD_NAME", "DOWNLOAD_NO_EXTRACT", "DOWNLOAD_NO_PROGRESS", "TIMEOUT",
    "INACTIVITY_TIMEOUT", "HTTP_USERNAME", "HTTP_PASSWORD", "HTTP_HEADER",
    "TLS_VERIFY", "TLS_CAINFO", "NETRC", "NETRC_FILE", "GIT_REPOSITORY",
    "GIT_TAG", "GIT_REMOTE_NAME", "GIT_SUBMODULES", "GIT_SUBMODULES_RECURSE",
    "GIT_SHALLOW", "GIT_PROGRESS", "GIT_CONFIG", "GIT_REMOTE_UPDATE_STRATEGY",
    "SVN_REPOSITORY", "SVN_REVISION", "SVN_USERNAME", "SVN_PASSWORD",
    "SVN_TRUST_CERT", "HG_REPOSITORY", "HG_TAG", "CVS_REPOSITORY", "CVS_MODULE",
    "CVS_TAG", "UPDATE_COMMAND", "UPDATE_DISCONNECTED", "PATCH_COMMAND",
    "SOURCE_SUBDIR", "LOG_DOWNLOAD", "LOG_UPDATE", "LOG_PATCH",
    "LOG_MERGED_STDOUTERR", "LOG_OUTPUT_ON_FAILURE", "USES_TERMINAL_DOWNLOAD",
    "USES_TERMINAL_UPDATE", "USES_TERMINAL_PATCH", "DEPENDS",
    "EXCLUDE_FROM_ALL", "LIST_SEPARATOR"
]

EXTERNAL_PROJECT_ADD_OPTIONS = FETCH_CONTENT_DECLARE_OPTIONS + [
    "CONFIGURE_COMMAND", "CMAKE_COMMAND", "CMAKE_GENERATOR",
    "CMAKE_GENERATOR_PLATFORM", "CMAKE_GENERATOR_TOOLSET",
    "CMAKE_GENERATOR_INSTANCE", "CMAKE_ARGS", "CMAKE_CACHE_ARGS",
    "CMAKE_CACHE_DEFAULT_ARGS", "CONFIGURE_HANDLED_BY_BUILD", "BUILD_COMMAND",
    "BUILD_IN_SOURCE", "BUILD_ALWAYS", "BUILD_BYPRODUCTS", "INSTALL_COMMAND",
    "TEST_COMMAND", "TEST_BEFORE_INSTALL", "TEST_AFTER_INSTALL",
    "TEST_EXCLUDE_FROM_MAIN", "LOG_CONFIGURE", "LOG_BUILD", "LOG_INSTALL",
    "LOG_TEST", "USES_TERMINAL_CONFIGURE", "USES_TERMINAL_BUILD",
    "USES_TERMINAL_INSTALL", "USES_TERMINAL_TEST", "STEP_TARGETS",
    "INDEPENDENT_STEP_TARGETS", "LIST_SEPARATOR", "COMMAND"
]


def format_cmake_options(options: Dict[str, Any],
                         keys: Optional[List[str]] = None) -> str:
  """Formats a Dict as CMake options, optionally filtered to keys."""
  if not keys:
    keys = options.keys()
  entries = []
  first_str = -1  # track first string value for packing onto lines
  for k in keys:
    v = options.get(k, None)
    if v is None:
      v = options.get(k.lower(), None)
      if v is None:
        continue
    k = k.upper()

    if isinstance(v, list):
      if not v:
        continue
      if first_str == -1:
        first_str = len(entries)
      entries.append((k, " ".join(v)))
    elif isinstance(v, str):
      if first_str == -1:
        first_str = len(entries)
      if not v:
        v = '""'
      elif " " in v and v[0] != '"':
        v = '"{v}"'.format(v=v)
      entries.append((k, v))
    elif v:
      entries.append((k, None))

  extra = ""
  for i in range(0, len(entries)):
    t = entries[i]
    if first_str == -1 or i < first_str:
      extra += f" {t[0]}"
      continue
    if isinstance(t[1], str):
      extra += f"\n  {t[0]: <10} {t[1]}"
    else:
      extra += f"\n  {t[0]}"
  return extra


class CMakeScriptBuilder:
  """Assists in building CMakeLists.txt files."""
  INDENT = "\n    "
  PART = """  %s
    %s
"""

  def __init__(self):
    self._includes: Set[str] = set()
    self._sections: Dict[int, List[str]] = collections.defaultdict(lambda: [])
    self._subdirs: List[str] = []
    self._default_section = 1000

  @property
  def default_section(self) -> int:
    return self._default_section

  def set_default_section(self, section: int):
    self._default_section = section

  def as_text(self) -> str:
    sections = []
    for k in sorted(set(self._sections.keys())):
      sections.extend(self._sections[k])

    subdirs = [f"add_subdirectory({x})\n" for x in self._subdirs]
    return "".join(sorted(self._includes) + sections + subdirs)

  def addtext(self, text: str, section: Optional[int] = None):
    """Adds raw text to the cmake file."""
    if not section:
      section = self.default_section
    self._sections[section].append(text)

  def set(self,
          variable: str,
          value: str,
          scope: str = "",
          section: Optional[int] = None):
    # https://cmake.org/cmake/help/latest/command/set.html
    if scope == "FORCE" or scope == "CACHE":
      force = "FORCE" if scope == "FORCE" else ""
      self.addtext(
          f'set({variable: <12} {value} CACHE INTERNAL "" {force})\n',
          section=section)
    else:
      if scope != "PARENT_SCOPE":
        scope = ""
      self.addtext(f"set({variable: <12} {value} {scope})\n", section=section)

  def add_subdirectory(self,
                       source_dir: str,
                       binary_dir: Optional[str] = None,
                       exclude_from_all: bool = False):
    # https://cmake.org/cmake/help/latest/command/add_subdirectory.html
    if not binary_dir:
      binary_dir = ""
    self._subdirs.append(" ".join(
        filter(None, [
            source_dir, binary_dir,
            "EXCLUDE_FROM_ALL" if exclude_from_all else ""
        ])))

  def find_package(
      self,  #
      name: str,
      version: Optional[str],
      options: Dict[str, Any],
      section: Optional[int] = None):
    # https://cmake.org/cmake/help/latest/command/find_package.html
    if not version:
      version = ""
    extra = format_cmake_options(options, FIND_PACKAGE_OPTIONS)
    self.addtext(
        "find_package({x})\n".format(
            x=" ".join(filter(None, [name, version, extra]))),
        section=section)

  def fetch_content_make_available(self,
                                   name: str,
                                   section: Optional[int] = None):
    # https://cmake.org/cmake/help/latest/module/FetchContent.html
    # https://github.com/Kitware/CMake/blob/master/Modules/FetchContent.cmake
    self._includes.add("include(FetchContent)\n")
    self.addtext(f"FetchContent_MakeAvailable({name})\n", section=section)

  def fetch_content_declare(self,
                            name: str,
                            options: Dict[str, Any],
                            section: Optional[int] = None):
    # https://cmake.org/cmake/help/latest/module/FetchContent.html
    self._includes.add("include(FetchContent)\n")
    extra = format_cmake_options(options, FETCH_CONTENT_DECLARE_OPTIONS)
    self.addtext(
        f"""
FetchContent_Declare(
  {name}{extra})
""", section=section)

  def cc_test(self, name: str, srcs: Set[str], deps: Set[str]):
    self.addtext("""\n
tensorstore_cc_test(
  NAME
    %s
  SRCS
    %s
  COPTS
    $\x7bTENSORSTORE_TEST_COPTS\x7d
  DEPS
    %s
)
""" % (name, self.INDENT.join(sorted(srcs)), self.INDENT.join(sorted(deps))))

  def cc_library(self, name: str, srcs: Set[str], hdrs: Set[str],
                 deps: Set[str], is_public: bool):
    rest = ""
    if srcs:
      rest += self.PART % ("SRCS", self.INDENT.join(sorted(srcs)))
    if hdrs:
      rest += self.PART % ("HDRS", self.INDENT.join(sorted(hdrs)))
    if deps:
      rest += self.PART % ("DEPS", self.INDENT.join(sorted(deps)))
    if is_public:
      rest += "  PUBLIC"

    self.addtext("""\n
tensorstore_cc_library(
  NAME
    %s
  COPTS
    $\x7bTENSORSTORE_DEFAULT_COPTS\x7d
  LINKOPTS
    $\x7bTENSORSTORE_DEFAULT_LINKOPTS\x7d
%s
)
""" % (name, rest))


def is_visibility_public(vis: List[str]) -> bool:
  return "//visibility:public" in vis


def format_project_target(project: str, target: str) -> str:
  """Format a target name in a project as project::target_name."""
  if len(target) > 1 and target[0] == project:
    f = target[1:]
  else:
    f = target
  if not f:
    return None
  return "{project}::{target}".format(project=project, target="_".join(f))


def canonical_bazel_target(target: str) -> str:
  """Returns the canonical bazel target name."""
  a = target.rfind(":")
  b = target.rfind("/")
  if a > b:
    return target
  suffix = target[b + 1:]
  return f"{target}:{suffix}"


def bazel_target_to_path(
    target: str, path_elements: Optional[List[str]] = None) -> List[str]:
  """Format bazel file target as a list of path elements."""
  # target is rooted at the base of the repo.
  if not path_elements:
    path_elements = []
  paths = list(filter(None, re.split("[:/]+", target)))
  if target.startswith("//"):  # root label
    return paths
  else:  # local label
    return path_elements + paths


class Converter:
  FIRST_SECTION = 100

  def __init__(self, project):
    self.project = project
    self.workspace = ""
    self.errors = []
    self.dep_mapping = {}
    # transient state
    self._filename = ""
    self._path_elements = []
    self._prefix_len = 0
    self._default_visibility = []
    self._builder = None
    self._all_deps = set()
    self._proto_libraries = dict()

  def set_filename(self, filename: str, max_prefix: int = KEY_DEPTH):
    print(f"Processing {filename}")
    self._default_visibility = []
    self._filename = filename
    self._path_elements = self._filename.split("/")[:-1]  # remove build.
    if self.project in self._path_elements:
      # begins with self.project
      idx = self._path_elements.index(self.project)
      self._path_elements = self._path_elements[idx:]
    self._prefix_len = max_prefix
    if len(self._path_elements) < max_prefix:
      self._prefix_len = len(self._path_elements)

  def get_key_from_filename(self) -> Tuple[str, ...]:
    return tuple(self._path_elements[:self._prefix_len])

  def set_builder(self, builder: CMakeScriptBuilder):
    self._builder = builder

  def set_default_visibility(self, vis: List[str]):
    self._default_visibility = vis

  @property
  def default_section(self) -> int:
    if not self._builder:
      return 0
    return self._builder.default_section

  def set_default_section(self, section: int):
    self._builder.set_default_section(section)

  @property
  def filename(self) -> str:
    return self._filename

  def _is_relative_path(self, path: List[str]) -> bool:
    if len(path) < self._prefix_len:
      return False
    for i in range(self._prefix_len):
      if path[i] != self._path_elements[i]:
        return False
    return True

  def _get_name(self, kwargs: Dict[str, Any]) -> str:
    """Format kwargs[name] as a cmake target name."""
    name = self._path_elements + [kwargs.get("name")]
    if name[0] == self.project:
      return "_".join(name[1:])
    return "_".join(name)

  def _map_single_file(self, file: str) -> str:
    """Format bazel file target as a path."""
    paths = bazel_target_to_path(file, self._path_elements)
    if self._is_relative_path(paths):
      return "/".join(paths[self._prefix_len:])
    return "/".join(["${CMAKE_SOURCE_DIR}"] + paths)

  def _map_files(self, files: List[str]) -> Set[str]:
    return set(filter(None, [self._map_single_file(x) for x in files]))

  def _map_single_dep(self, dep: str) -> str:
    dep = canonical_bazel_target(dep)
    if dep in self.dep_mapping:
      return self.dep_mapping[dep]

    # handle mappings in the current path.
    if dep.startswith(":"):
      f = format_project_target(self.project, self._path_elements + [dep[1:]])
      if f:
        return f

    # handle self mappings
    if dep.startswith("f@{self.workspace}") or dep.startswith("//"):
      f = format_project_target(self.project,
                                re.split("[:/]+",
                                         dep.split("//")[1]))
      if f:
        return f

    # not handled.
    self.errors.append(f"Missing mapping for {dep} in {self._filename}")
    return dep

  def _get_deps(self, kwargs: Dict[str, Any]) -> Set[str]:
    return set(
        filter(None, [self._map_single_dep(x) for x in kwargs.get("deps", [])]))

  def _adapt_to_package_options(self, kwargs: Dict[str, Any]):
    options = kwargs
    urls = options.get("urls", [])
    if urls:
      del options["urls"]
      options.update({"URL": urls[0]})
    sha256 = kwargs.get("sha256", None)
    if sha256:
      del options["sha256"]
      options.update({"URL_HASH": f"SHA256={sha256}"})

    # TODO: Enable patching by default.
    # Apparently this attempts to patch -populate
    if kwargs.get("allow_patch", False):
      patch_commands = []
      for x in kwargs.get("patches", []):
        patch_commands.append(
            "patch < " + "/".join(["${CMAKE_SOURCE_DIR}"] +
                                  bazel_target_to_path(x, self._path_elements)))
      if patch_commands:
        options.update({"PATCH_COMMAND": " && ".join(patch_commands)})

    # strip_prefix = kwargs.get("strip_prefix", "")
    return options

  def addtext(self, text: str):
    """Adds raw content to the CMakeLists.txt."""
    if text:
      self._builder.addtext(text)

  def add_todo(self, kwargs: Dict[str, Any]):
    """Generates a TODO comment in the CMakeLists.txt."""
    kind = sys._getframe(1).f_code.co_name  # annotate with rule name
    name = self._get_name(kwargs)
    self.addtext("\n# TODO: %s %s\n" % (kind, name))

  def add_find_package(self, name: str, version: str, fallback: bool,
                       kwargs: Dict[str, Any]):
    is_required = not fallback
    if not is_required:
      urls = kwargs.get("urls", [])
      if not urls:
        is_required = True
    self._builder.addtext("\n")

    myargs = dict()
    myargs.update(kwargs)
    if is_required:
      myargs.update({"REQUIRED": True})
    self._builder.find_package(name, version, myargs)

    if not is_required:
      self._builder.addtext("if(NOT ${%s_FOUND})\n" % name)
      innername = kwargs.get("name", name)
      self._builder.fetch_content_declare(
          innername, self._adapt_to_package_options(kwargs))
      self._builder.addtext(f"FetchContent_MakeAvailable({innername})\n")
      self._builder.addtext("endif()\n\n")

  def add_fetch_content(self, name, kwargs: Dict[str, Any]):
    urls = kwargs.get("urls", [])
    if not urls:
      return
    self._builder.addtext("\n")
    self._builder.fetch_content_declare(name,
                                        self._adapt_to_package_options(kwargs))
    make_available = kwargs.get("make_available", True)
    if make_available:
      self._builder.fetch_content_make_available(name)

  def add_settings(self, settings: List):
    for env in settings:
      try:
        self._builder.set(env[0], env[1], scope="FORCE")
      except:
        self.errors.append(f"Failed to set {env} in {self._filename}")

  def add_cc_test(self, **kwargs):
    """Generates a tensorstore_cc_test."""
    name = self._get_name(kwargs)
    srcs = self._map_files(kwargs.get("srcs", []) + kwargs.get("hdrs", []))
    deps = self._get_deps(kwargs)
    self._all_deps.update(deps)

    if not srcs:
      self._builder.addtext(f"# Missing {name}\n")
    else:
      self._builder.cc_test(name, srcs, deps)

  def add_cc_library(self, **kwargs):
    """Generates a tensorstore_cc_library."""
    name = self._get_name(kwargs)
    deps = self._get_deps(kwargs)
    self._all_deps.update(set(deps))
    srcs = self._map_files(kwargs.get("srcs", []))
    hdrs = self._map_files(kwargs.get("hdrs", []))
    is_public = is_visibility_public(
        kwargs.get("visibility", self._default_visibility))
    self._builder.cc_library(name, srcs, hdrs, deps, is_public)

  def first_pass_proto_library(self, kind: str, kwargs: Dict[str, Any]):
    name = self._get_name(kwargs)
    if not name:
      return
    name = f"{self.project}::{name}"
    deps = self._get_deps(kwargs)
    if kind == "proto":
      # Forward mapping
      srcs = self._map_files(kwargs.get("srcs", []))
      self._proto_libraries[(kind, name)] = (srcs, deps)
      return
    for x in deps:
      # Reverse mapping
      self._proto_libraries[(kind, x)] = name

  def add_cc_proto_library(self, kwargs):
    # All the proto_libraries should be registered at this point,
    # So it should be possible to extract proto library mappings.
    name = self._get_name(kwargs)
    if not name:
      return

    protos = set()
    deps = set()
    for x in self._get_deps(kwargs):
      t = self._proto_libraries[("proto", x)]
      protos.update(t[0])  # srcs
      for y in t[1]:  # deps
        k = ("cc", y)
        if k in self._proto_libraries:
          deps.add(self._proto_libraries[k])

    if deps:
      deps = "\n  DEPS\n    %s" % CMakeScriptBuilder.INDENT.join(sorted(deps))
    else:
      deps = ""

    self._builder.addtext("""\n
tensorstore_proto_cc_library(
  NAME
    %s
  PROTOS
    %s%s
  COPTS
    $\x7bTENSORSTORE_DEFAULT_COPTS\x7d
  LINKOPTS
    $\x7bTENSORSTORE_DEFAULT_LINKOPTS\x7d
  PUBLIC
)
""" % (name, CMakeScriptBuilder.INDENT.join(sorted(protos)), deps))


class BazelGlobalsDict(dict):

  def __init__(self):
    pass

  def __setitem__(self, key, val):
    if not hasattr(self, key):
      dict.__setitem__(self, key, val)

  def __getitem__(self, key):
    if hasattr(self, key):
      return getattr(self, key)
    if dict.__contains__(self, key):
      return dict.__getitem__(self, key)
    return self._unimplemented

  def _unimplemented(self, *args, **kwargs):
    pass

  def glob(self, *args, **kwargs):
    # NOTE: Non-trivial uses of glob() in BUILD files will need attention.
    return []

  def select(self, arg_dict):
    return []

  def load(self, *args):
    pass

  def package_name(self, **kwargs):
    return ""


class BuildFileFunctions(BazelGlobalsDict):
  """Globals dict for exec('**/BUILD')."""

  def __init__(self, converter):
    super(BuildFileFunctions, self).__init__()
    self.converter = converter
    self.converter.addtext(f"\n# From {self.converter.filename}\n\n")

  def package(self, **kwargs):
    self.converter.set_default_visibility(kwargs.get("default_visibility", []))

  def cc_library(self, **kwargs):
    self.converter.add_cc_library(**kwargs)

  def cc_test(self, **kwargs):
    self.converter.add_cc_test(**kwargs)

  def cc_library_with_strip_include_prefix(self, **kwargs):
    self.converter.add_cc_library(**kwargs)

  def cc_with_non_compile_test(self, **kwargs):
    self.converter.add_todo(kwargs)

  def pybind11_cc_library(self, **kwargs):
    self.converter.add_cc_library(**kwargs)

  def pybind11_py_extension(self, **kwargs):
    self.converter.add_cc_library(**kwargs)

  def tensorstore_cc_binary(self, **kwargs):
    self.converter.add_todo(kwargs)

  def tensorstore_cc_library(self, **kwargs):
    self.converter.add_cc_library(**kwargs)

  def tensorstore_cc_test(self, **kwargs):
    self.converter.add_cc_test(**kwargs)

  def tensorstore_cc_proto_library(self, **kwargs):
    self.converter.add_cc_proto_library(kwargs)


class BuildFileFirstPass(BazelGlobalsDict):
  """First pass globasl dict for exec('**/BUILD'), which registers proto libraries."""

  def __init__(self, converter):
    super(BuildFileFirstPass, self).__init__()
    self.converter = converter

  def tensorstore_proto_library(self, **kwargs):
    self.converter.first_pass_proto_library("proto", kwargs)

  def proto_library(self, **kwargs):
    self.converter.first_pass_proto_library("proto", kwargs)

  def tensorstore_cc_proto_library(self, **kwargs):
    self.converter.first_pass_proto_library("cc", kwargs)

  def cc_proto_library(self, **kwargs):
    self.converter.first_pass_proto_library("cc", kwargs)


class WorkspaceFileFunctions(BazelGlobalsDict):
  """Globals dict for exec('WORKSPACE') and exec('workspace.bzl')."""

  def __init__(self, converter):
    super(WorkspaceFileFunctions, self).__init__()
    self.converter = converter
    self._initial_comment_added = False

  def _add_file_comment(self):
    if self._initial_comment_added:
      return
    self._initial_comment_added = True
    self.converter.addtext(f"\n# From {self.converter.filename}\n")

  def third_party_http_archive(self):
    pass

  def maybe(self, fn, **kwargs):
    pass

  def workspace(self, **kwargs):
    self.converter.workspace = kwargs.get("name", "")

  def cmake_add_dep_mapping(self, **kwargs):
    # stores the mapping from "canonical" bazel target to cmake target.
    target_mapping = kwargs.get("target_mapping", {})
    for k, v in target_mapping.items():
      self.converter.dep_mapping[canonical_bazel_target(k)] = v

  def cmake_set_section(self, **kwargs):
    section = kwargs.get("section", None)
    if section:
      self.converter.set_default_section(section)

  def cmake_get_section(self) -> int:
    return self.converter.default_section

  def cmake_raw(self, **kwargs):
    self._add_file_comment()
    text = kwargs.get("text", "")
    self.converter.addtext(text)

  def cmake_find_package(self, **kwargs):
    self._add_file_comment()
    name = kwargs.get("name", None)
    version = kwargs.get("version", None)
    fallback = kwargs.get("fallback", False)
    settings = kwargs.get("settings", None)
    if not name:
      return
    if settings:
      self.converter.add_settings(settings)
    myargs = dict()
    myargs.update(kwargs)

    def maybe(self, fn, **kwargs):
      nonlocal name
      nonlocal version
      nonlocal myargs
      nonlocal fallback
      myargs.update(kwargs)
      self.converter.add_find_package(name, version, fallback, myargs)

    tmp = self.maybe
    self.maybe = types.MethodType(maybe, self)
    self["repo"]()
    self.maybe = tmp

  def cmake_fetch_content_package(self, **kwargs):
    self._add_file_comment()
    name = kwargs.get("name", None)
    settings = kwargs.get("settings", None)
    if not name:
      return
    if settings:
      self.converter.add_settings(settings)
    myargs = dict()
    myargs.update(kwargs)

    def maybe(self, fn, **kwargs):
      nonlocal name
      nonlocal myargs
      myargs.update(kwargs)
      self.converter.add_fetch_content(name, myargs)

    tmp = self.maybe
    self.maybe = types.MethodType(maybe, self)
    self["repo"]()
    self.maybe = tmp


class CMakeListSet:
  """Maintains a set of CMakeScriptBuilder keyed by tuple."""

  def __init__(self):
    self.cmakelists = dict()

  def get_script_builder(self, key: Tuple[str, ...]) -> CMakeScriptBuilder:
    """Adds an entry to the CMakeLists for the current key.

     The key is assumed to be a hierarchical tuple of directories.
     """
    if len(key) > 1:
      if key not in self.cmakelists:
        self.get_script_builder(key[:-1]).add_subdirectory(key[-1])
    if key not in self.cmakelists:
      self.cmakelists[key] = CMakeScriptBuilder()
    return self.cmakelists[key]

  def generate_files(self):
    """Generate CMakeLists.txt files for the project."""
    print(f"Generating CMakeLists.txt")

    # Generate the top-level CMakeLists.txt file from the template in the
    # CMake directory.
    third_party_template = ""
    add_subdir_template = set()
    for k, v in self.cmakelists.items():
      if k[0] != "third_party":
        add_subdir_template.add(f"add_subdirectory({k[0]})\n")
        continue
      third_party_template += v.as_text()

    template = open("CMake/CMakeLists.template").read()
    template = template.replace("{third_party_template}", third_party_template)
    template = template.replace("{add_subdir_template}",
                                "".join(sorted(add_subdir_template)))
    with open("CMakeLists.txt", "w") as f:
      f.write(template)

    # Generate subdirectory CMakeLists.txt from the BUILD content.
    for k, v in self.cmakelists.items():
      if k[0] == "third_party":
        continue
      filename = os.path.join(*(list(k) + ["CMakeLists.txt"]))
      text = ("# Autogenerated by bazel_to_cmake.py\n\n" + v.as_text())
      print(f"Generating {filename}")
      with open(filename, "w") as f:
        f.write(text)


def main():
  """Recursively process BUILD and workspace.bzl to generate CMakeLists.txt."""

  converter = Converter("tensorstore")

  # Python headers have no current link target.
  converter.dep_mapping["@local_config_python//:python_headers"] = ""

  # Initial configuration belonga to third_party/CMakeLists.txt
  # This is specified by the key tuple ('third_party')
  section = 1000
  cmakelists = CMakeListSet()
  builder = cmakelists.get_script_builder(tuple(["third_party"]))
  converter.set_builder(builder)

  # Collect WORKSPACE and workspace.bzl files.
  workspace_files = ["WORKSPACE"] + sorted(
      set(glob.glob("third_party/**/workspace.bzl", recursive=True)))

  # Process third_party workspace.bzl files to add cmake mappings.
  for workspace in workspace_files:
    converter.set_filename(workspace)
    builder.set_default_section(section)
    section += 1000
    exec(open(converter.filename).read(), WorkspaceFileFunctions(converter))

  # Process BUILD files to add cmake mappings. Do this in two passes, so the
  # first pass can record dependency information used by protos.
  build_files = []
  for build in sorted(set(glob.glob("**/BUILD", recursive=True))):
    if (build.find("tensorstore/") == -1 or build.find("examples/") >= 0 or
        build.find("docs/") >= 0 or build.find("python/") >= 0 or
        build.find("third_party/") >= 0):
      continue
    converter.set_filename(build)
    my_builder = cmakelists.get_script_builder(
        converter.get_key_from_filename())
    converter.set_builder(my_builder)
    # update section
    section = my_builder.default_section + 1000
    my_builder.set_default_section(section)
    build_files.append((build, section))
    exec(open(converter.filename).read(), BuildFileFirstPass(converter))

  # Second pass. Uses section from build_files to ensure consistency
  for build, section in build_files:
    converter.set_filename(build)
    my_builder = cmakelists.get_script_builder(
        converter.get_key_from_filename())
    my_builder.set_default_section(section)
    converter.set_builder(my_builder)
    exec(open(converter.filename).read(), BuildFileFunctions(converter))

  if converter.errors:
    print("\n".join(converter.errors))
    return 1

  cmakelists.generate_files()
  return 0


if __name__ == "__main__":
  sys.exit(main())
