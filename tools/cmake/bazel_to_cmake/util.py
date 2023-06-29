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
"""Miscellaneous utility functions."""

import glob
import os
import pathlib
import re
from typing import List, Optional, Set

from .starlark.bazel_glob import glob_pattern_to_regexp


# Unfortunately, pathlib.PurePath.is_relative_to is a python3.9 invention.
def is_relative_to(
    left: pathlib.PurePath, right: pathlib.PurePath, _use_attr: bool = True
) -> bool:
  """Return True if the path is relative to another path or False."""
  if _use_attr and hasattr(left, "is_relative_to"):
    return left.is_relative_to(right)
  other = type(left)(right)
  return other == left or other in left.parents


def write_file_if_not_already_equal(path: pathlib.PurePath, content: bytes):
  """Ensures `path` contains `content`.

  Does not update the modification time of `path` if it already contains
  `content`, to avoid unnecessary rebuilding.

  Args:
    path: Path to file.
    content: Content to write.
  """
  try:
    if pathlib.Path(path).read_bytes() == content:
      # Only write if it does not already have the desired contents, to avoid
      # unnecessary rebuilds.
      return
  except FileNotFoundError:
    pass
  os.makedirs(path.parent, exist_ok=True)
  pathlib.Path(path).write_bytes(content)


# https://cmake.org/cmake/help/latest/command/if.html#basic-expressions
# CMake considers any of the following a "false constant":
# - "0"
# - "OFF" (case insensitive)
# - "N" (case insensitive)
# - "NO" (case insensitive)
# - ""
# - "IGNORE" (case insensitive)
# - "NOTFOUND" (case insensitive)
# - ending in "-NOTFOUND"
_CMAKE_FALSE_PATTERN = re.compile(
    r"0|[oO][fF][fF]|[nN][oO]|[fF][aA][lL][sS][eE]|[nN]|[iI][gG][nN][oO][rR][eE]|NOTFOUND||.*-NOTFOUND",
    re.DOTALL,
)


def cmake_is_true(value: Optional[str]) -> bool:
  """Determines if a string is considered by CMake to be TRUE."""
  if value is None:
    return False
  return not _CMAKE_FALSE_PATTERN.fullmatch(value)


def cmake_is_windows(value: Optional[str]) -> bool:
  """Determines if a string is considered by CMake to be TRUE."""
  if value is None:
    return False
  return value.startswith("Windows")


def cmake_logging_verbose_level(value: Optional[str]) -> int:
  """Returns the logging verbosity level based on CMAKE_MESSAGE_LOG_LEVEL."""
  if value is None:
    return 0
  value = value.lower()
  if "verbose" in value:
    return 1
  elif "debug" in value:
    return 2
  elif "trace" in value:
    return 3
  else:
    return 0


def _get_build_patterns(package_patterns: List[str]):
  patterns = []
  for package_pattern in package_patterns:
    pattern = package_pattern
    if pattern:
      pattern += "/"
    patterns.append(pattern + "BUILD")
    patterns.append(pattern + "BUILD.bazel")
  return patterns


def get_matching_build_files(
    root_dir: pathlib.PurePath,
    include_packages: List[str],
    exclude_packages: List[str],
) -> List[str]:
  """Returns the relative path of matching BUILD files.

  Args:
    root_dir: Path to root directory for the repository.
    include_packages: List of glob patterns matching package directory names to
      include.  May use "*" and "**".  For example, `["tensorstore/**"]`.
    exclude_packages: List of glob patterns matching package directory names to
      exclude.

  Returns:
    Sorted list of matching build files, relative to `root_dir`.
  """
  if isinstance(root_dir, pathlib.PureWindowsPath):
    root_dir = pathlib.PurePath(root_dir.as_posix())
  root_prefix = root_dir.as_posix() + "/"

  include_patterns = _get_build_patterns(include_packages)
  exclude_regexp = re.compile(
      "(?:"
      + "|".join(
          glob_pattern_to_regexp(pattern)
          for pattern in _get_build_patterns(exclude_packages)
      )
      + ")"
  )

  build_file_set: Set[str] = set()
  for pattern in include_patterns:
    for build_filename in glob.iglob(root_prefix + pattern, recursive=True):
      path = pathlib.PurePath(build_filename)
      if not pathlib.Path(path).is_file():
        continue
      assert is_relative_to(path, root_dir)
      relative_path = path.relative_to(root_dir)
      if exclude_regexp.fullmatch(relative_path.as_posix()):
        continue
      build_file_set.add(relative_path.as_posix())
  return list(sorted(build_file_set))
