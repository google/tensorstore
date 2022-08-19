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
"""Utilities to aid in building CMakeLists.txt files."""

import collections
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

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


def kwargs_to_fetch_content_options(kwargs: Dict[str, Any],
                                    path_elements: Optional[List[str]] = None):
  """Convert bazel kwargs to options used by CMake FetchContent."""
  options = kwargs
  urls = options.get("urls", [])
  if urls:
    del options["urls"]
    options.update({"URL": urls[0]})
  sha256 = kwargs.get("sha256", None)
  if sha256:
    del options["sha256"]
    options.update({"URL_HASH": f"SHA256={sha256}"})
  # strip_prefix = kwargs.get("strip_prefix", "")

  # TODO: Enable patching by default.
  # Apparently this attempts to patch -populate
  path_elements = kwargs.get("path_elements", [])
  if kwargs.get("allow_patch", False):
    patch_commands = []
    for x in kwargs.get("patches", []):
      patch_commands.append("patch < " +
                            "/".join(["${CMAKE_SOURCE_DIR}"] +
                                     bazel_target_to_path(x, path_elements)))
    if patch_commands:
      options.update({"PATCH_COMMAND": " && ".join(patch_commands)})
  return options


class CMakeBuilder:
  """Utility to assist in building a CMakeLists.txt file.

  CMakeBuilder represents one CMakeLists.txt file, and has methods to
  add common script expressions corresponding to Bazel files.
"""
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
    self.addtext(f"FetchContent_MakeAvailable({name})\n\n", section=section)

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

  def cc_proto_library(self, name: str, protos: Set[str], deps: Set[str]):
    if deps:
      deps = "\n  DEPS\n    %s" % self.INDENT.join(sorted(deps))
    else:
      deps = ""

    self.addtext("""\n
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
""" % (name, self.INDENT.join(sorted(protos)), deps))


class CMakeListSet:
  """Maintains a set of CMakeBuilder keyed by a path tuple."""

  def __init__(self):
    self.cmakelists = dict()

  def get_root_script_builder(self) -> CMakeBuilder:
    return self.get_script_builder(tuple(["_ROOT_"]))

  def get_script_builder(self, key: Tuple[str, ...]) -> CMakeBuilder:
    """Adds an entry to the CMakeLists for the current key.

     The key is assumed to be a hierarchical tuple of directories.
     """
    if len(key) > 1:
      if key not in self.cmakelists:
        self.get_script_builder(key[:-1]).add_subdirectory(key[-1])
    if key not in self.cmakelists:
      self.cmakelists[key] = CMakeBuilder()
    return self.cmakelists[key]

  def generate_files(self):
    """Generate CMakeLists.txt files for the project."""
    print(f"Generating CMakeLists.txt")

    # Generate the top-level CMakeLists.txt file from the template in the
    # CMake directory.
    root_template = ""
    add_subdir_template = set()
    for k, v in self.cmakelists.items():
      if k[0] != "_ROOT_":
        add_subdir_template.add(f"add_subdirectory({k[0]})\n")
        continue
      root_template += v.as_text()

    template = open("CMake/CMakeLists.template").read()
    template = template.replace("{root_template}", root_template)
    template = template.replace("{add_subdir_template}",
                                "".join(sorted(add_subdir_template)))
    with open("CMakeLists.txt", "w") as f:
      f.write(template)

    # Generate subdirectory CMakeLists.txt from the BUILD content.
    for k, v in self.cmakelists.items():
      if k[0] == "_ROOT_":
        continue
      filename = os.path.join(*(list(k) + ["CMakeLists.txt"]))
      text = ("# Autogenerated by bazel_to_cmake.py\n\n" + v.as_text())
      print(f"Generating {filename}")
      with open(filename, "w") as f:
        f.write(text)
