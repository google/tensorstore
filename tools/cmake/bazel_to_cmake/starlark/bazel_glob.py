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
"""bazel glob function."""

# pylint: disable=missing-function-docstring

import glob as _glob
import os
import pathlib
import re

from typing import List, Optional, Dict, Set


def glob_pattern_to_regexp(glob_pattern: str) -> str:
  """Computes a regular expression for a (recursive) glob pattern.

  This is used to efficiently evaluate exclusion criteria.

  Args:
    glob_pattern: Glob pattern with "*" and "**".  Note that "[]" is not
      supported.

  Returns:
    Corresponding regular expression.
  """
  regexp_parts = []

  for i, part in enumerate(glob_pattern.split("/")):
    sep_prefix = "/" if i > 0 else ""
    if part == "**":
      regexp_parts.append(f"(?:{sep_prefix}.*)?")
      continue
    regexp_parts.append(sep_prefix)
    for x in re.split(r"(\*)", part):
      if x == "*":
        regexp_parts.append("[^/]*")
      else:
        regexp_parts.append(re.escape(x))
  return "".join(regexp_parts)


def glob(directory: str,
         include: List[str],
         exclude: Optional[List[str]] = None,
         allow_empty: bool = True) -> List[str]:

  exclude_regexp = None
  if exclude:
    exclude_regexp = re.compile("(?:" + "|".join(
        glob_pattern_to_regexp(pattern) for pattern in exclude) + ")")

  is_subpackage_dir: Dict[str, bool] = {}
  is_subpackage_dir[""] = False

  # Exclude files in subdirectory packages; anything with a 'BUILD' file
  # counts, so search all directories down to the actual file.
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
      if result is None:
        # These build files haven't been checked yet; check them
        build_path = str(
            pathlib.PurePosixPath(directory).joinpath(subdir, "BUILD"))
        result = (
            os.path.exists(build_path) or os.path.exists(build_path + ".bazel"))
        is_subpackage_dir[subdir] = result
      if result:
        return True
    return False

  def get_matches(pattern: str):
    for match in _glob.iglob(os.path.join(directory, pattern), recursive=True):
      if not os.path.isfile(match):
        continue
      relative = os.path.relpath(match, directory)
      if os.sep != "/":
        relative = relative.replace(os.sep, "/")
      if in_subpackage(relative):
        continue
      if exclude_regexp is not None and exclude_regexp.fullmatch(relative):
        continue
      yield relative

  matches: Set[str] = set()
  for pattern in include:
    matches.update(get_matches(pattern))

  if not matches and not allow_empty:
    raise ValueError("glob produced empty result list")
  return sorted(matches)
