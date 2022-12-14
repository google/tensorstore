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

# pylint: disable=g-doc-args,g-doc-return-or-yield

import collections
import json
import os
from typing import Dict, List, Optional, Set, Tuple, Sequence


def quote_string(x: str) -> str:
  """Quotes a string for CMake."""
  return json.dumps(x)


def quote_path(x: str) -> str:
  """Quotes a path, converting backslashes to forward slashes.

  While CMake in some cases allows backslashes to be escaped, in other cases
  paths are passed without escaping.  Using forward slashes reduces the risk of
  problems.
  """
  if os.sep != "/":
    x = x.replace(os.sep, "/")
  return quote_string(x)


def quote_list(y: Sequence[str], separator: str = " ") -> str:
  return separator.join(quote_string(x) for x in y)


def quote_path_list(y: Sequence[str], separator: str = " ") -> str:
  return separator.join(quote_path(x) for x in y)


INCLUDE_SECTION = 0
FIND_PACKAGE_SECTION = 1
OPTIONS_SECTION = 10
ENABLE_LANGUAGES_SECTION = 11
LOCAL_MIRROR_DOWNLOAD_SECTION = 20
FETCH_CONTENT_DECLARE_SECTION = 30
FETCH_CONTENT_MAKE_AVAILABLE_SECTION = 40
FIND_DEP_PACKAGE_SECTION = 50


class CMakeBuilder:
  """Utility to assist in building a CMakeLists.txt file.

  CMakeBuilder represents one CMakeLists.txt file, and has methods to
  add common script expressions corresponding to Bazel files.
  """

  def __init__(self):
    self._sections: Dict[int, List[str]] = collections.defaultdict(lambda: [])
    self._unique: Set[Tuple[int, str]] = set()
    self._default_section = 1000

  def include(self, name):
    self.addtext(f"include({name})\n", section=INCLUDE_SECTION, unique=True)

  def find_package(self, name, section=FIND_PACKAGE_SECTION):
    self.addtext(
        f"find_package({name} REQUIRED)\n", section=section, unique=True)

  @property
  def default_section(self) -> int:
    return self._default_section

  def set_default_section(self, section: int):
    self._default_section = section

  def as_text(self) -> str:
    sections = []
    for k in sorted(set(self._sections.keys())):
      sections.extend(self._sections[k])

    return "".join(sections)

  def addtext(self,
              text: str,
              section: Optional[int] = None,
              unique: bool = False):
    """Adds raw text to the cmake file."""
    if section is None:
      section = self.default_section
    if unique:
      key = (section, text)
      if key in self._unique:
        return
      # FIND_PACKAGE / FIND_DEP_PACKAGE are special.
      if (section == FIND_DEP_PACKAGE_SECTION and
          (FIND_PACKAGE_SECTION, text) in self._unique):
        return
      self._unique.add(key)
    self._sections[section].append(text)

  def add_library_alias(
      self,
      target_name: str,
      alias_name: str,
      interface_only: bool = False,
      alwayslink: bool = False,
  ):
    """Generates an alias target with support for `alwayslink`."""
    alias_dest_name = target_name
    if alwayslink and not interface_only:
      alias_dest_name = f"{target_name}.alwayslink"
      self.addtext(f"""
add_library({alias_dest_name} INTERFACE)
if (BUILD_SHARED_LIBS)
  target_link_libraries({alias_dest_name} INTERFACE "$<LINK_LIBRARY:bazel_to_cmake_needed_library,{target_name}>")
else ()
  target_link_libraries({alias_dest_name} INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,{target_name}>")
endif()
""")
    self.addtext(f"add_library({alias_name} ALIAS {alias_dest_name})\n")
