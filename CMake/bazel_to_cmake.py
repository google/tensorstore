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

import glob
import re
import sys
import traceback
import types
from typing import Any, Dict, List, Set, Tuple

import cmake_builder

# Only generate CMakeLists.txt for targets up to this depth, by default.
# If the KEY_DEPTH is set to 1 (for tensorstore), the resulting file
# (tensorstore/CMakeLists.txt) is around 12k lines long.
KEY_DEPTH = 2


def is_visibility_public(vis: List[str]) -> bool:
  return "//visibility:public" in vis


class Converter:
  """Converter aids in converting between bazel BUILD and CMakeLists.txt"""

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

  def set_builder(self, builder: cmake_builder.CMakeBuilder):
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
    paths = cmake_builder.bazel_target_to_path(file, self._path_elements)
    if self._is_relative_path(paths):
      return "/".join(paths[self._prefix_len:])
    return "/".join(["${CMAKE_SOURCE_DIR}"] + paths)

  def _map_files(self, files: List[str]) -> Set[str]:
    return set(filter(None, [self._map_single_file(x) for x in files]))

  def _map_single_dep(self, dep: str) -> str:
    dep = cmake_builder.canonical_bazel_target(dep)
    if dep in self.dep_mapping:
      return self.dep_mapping[dep]

    # handle mappings in the current path.
    if dep.startswith(":"):
      f = cmake_builder.format_project_target(self.project,
                                              self._path_elements + [dep[1:]])
      if f:
        return f

    # handle self mappings
    if dep.startswith("f@{self.workspace}") or dep.startswith("//"):
      f = cmake_builder.format_project_target(
          self.project, re.split("[:/]+",
                                 dep.split("//")[1]))
      if f:
        return f

    # not handled.
    self.errors.append(f"Missing mapping for {dep} in {self._filename}")
    return dep

  def _get_deps(self, kwargs: Dict[str, Any]) -> Set[str]:
    return set(
        filter(None, [self._map_single_dep(x) for x in kwargs.get("deps", [])]))

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
          innername,
          cmake_builder.kwargs_to_fetch_content_options(kwargs,
                                                        self._path_elements))
      self._builder.addtext(f"FetchContent_MakeAvailable({innername})\n")
      self._builder.addtext("endif()\n\n")

  def add_fetch_content(self, name, kwargs: Dict[str, Any]):
    urls = kwargs.get("urls", [])
    if not urls:
      return
    self._builder.addtext("\n")
    self._builder.fetch_content_declare(
        name,
        cmake_builder.kwargs_to_fetch_content_options(kwargs,
                                                      self._path_elements))
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

    self._builder.cc_proto_library(name, protos, deps)


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
      self.converter.dep_mapping[cmake_builder.canonical_bazel_target(k)] = v

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


def main():
  """Recursively process BUILD and workspace.bzl to generate CMakeLists.txt."""

  cmakelists = cmake_builder.CMakeListSet()

  converter = Converter("tensorstore")

  # Python headers have no current link target.
  converter.dep_mapping["@local_config_python//:python_headers"] = ""

  # Initial configuration belongs to third_party/CMakeLists.txt
  section = 1000
  builder = cmakelists.get_root_script_builder()
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
