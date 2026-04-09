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

  def __init__(self) -> None:
    self._sections: dict[int, list[str]] = collections.defaultdict(list)
    self._unique: set[tuple[int, str]] = set()
    self._default_section = 1000

  def include(self, name: str) -> None:
    self.addtext(f"include({name})\n", section=INCLUDE_SECTION, unique=True)

  def find_package(
      self, name: str, section: int = FIND_PACKAGE_SECTION
  ) -> None:
    self.addtext(
        f"find_package({name} REQUIRED)\n", section=section, unique=True
    )

  @property
  def default_section(self) -> int:
    return self._default_section

  def set_default_section(self, section: int) -> None:
    self._default_section = section

  def as_text(self) -> str:
    sections = []
    for k in sorted(set(self._sections.keys())):
      sections.extend(self._sections[k])

    return "".join(sections)

  def addtext(
      self, text: str, section: int | None = None, unique: bool = False
  ) -> None:
    """Adds raw text to the cmake file."""
    if section is None:
      section = self.default_section
    if unique:
      key = (section, text)
      if key in self._unique:
        return
      # FIND_PACKAGE / FIND_DEP_PACKAGE are special.
      if (
          section == FIND_DEP_PACKAGE_SECTION
          and (FIND_PACKAGE_SECTION, text) in self._unique
      ):
        return
      self._unique.add(key)
    self._sections[section].append(text)
