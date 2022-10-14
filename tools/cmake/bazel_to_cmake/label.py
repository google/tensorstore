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
"""Defines data structures for Bazel labels."""

import re
from typing import NamedTuple, Union, Optional, Dict


class LabelLike:
  """Base class for `StarlarkLabel`, to avoid circular dependency."""

  def __str__(self) -> str:
    raise NotImplementedError


Label = str

# This is really intended to be `Union[str, StarlarkLabel]`, but that would
# introduce a circular type dependency.  This type is used for labels provided
# on the command-line, or by in BUILD files/bzl libraries.  Internally, they are
# always converted to resolved `Label` values before any further processing.
RelativeLabel = Union[str, LabelLike]

CMakeTarget = str


class ParsedLabel(NamedTuple):
  """Parsed label."""
  repo_name: str
  package_name: str
  target_name: str

  def __str__(self) -> str:
    return f"@{self.repo_name}//{self.package_name}:{self.target_name}"

  @property
  def package_and_target_name(self) -> str:
    return f"//{self.package_name}:{self.target_name}"


def parse_label(target: Union[Label, LabelLike]) -> ParsedLabel:
  target = str(target)
  m = re.fullmatch("@([^/]+)//([^:]*):(.*)$", target)
  if m is None:
    raise ValueError(f"Invalid bazel target: {target}")
  return ParsedLabel(m.group(1), m.group(2), m.group(3))


def label_to_generated_cmake_target(bazel_target: Label,
                                    cmake_project: str,
                                    alias: bool = False) -> CMakeTarget:
  """Computes the generated CMake target corresponding to a Bazel target."""
  bazel_target = str(bazel_target)

  # Strip off the workspace name
  m = re.fullmatch("^@([^/]+)//(.*)$", bazel_target)
  assert m is not None
  bazel_path = m.group(2)
  parts = [x for x in re.split("[:/]+", bazel_path) if x]
  if parts[0] == cmake_project and len(parts) > 1:
    parts = parts[1:]

  if len(parts) >= 2 and parts[-1] == parts[-2]:
    parts = parts[:-1]

  if alias:
    return cmake_project + "::" + "_".join(parts)
  return cmake_project + "_" + "_".join(parts)


def resolve_label(target: RelativeLabel,
                  repo_mapping: Optional[Dict[str, str]] = None,
                  base_package: Optional[str] = None) -> Label:
  """Returns the canonical bazel target name."""

  target = str(target)

  if (target.startswith("@") or target.startswith("//")) and ":" not in target:
    # Append last package component
    m = re.match(".*[@/]([^@/]+)$", target)
    if m is None:
      raise ValueError(f"Invalid bazel target: {target!r}")
    if "//" not in target:
      target = f"{target}//"
    target = f"{target}:{m.group(1)}"

  if not target.startswith("@"):
    # Relative target
    if base_package is None:
      raise ValueError(f"Bazel target must be absolute: {target!r}")

    m = re.fullmatch("@([^/]+)//([^:]*)$", base_package)
    if m is None:
      raise ValueError(f"Base package must be absolute: {base_package!r}")
    base_repo = m.group(1)
    base_package_name = m.group(2)
    if target.startswith("//"):
      target = f"@{base_repo}{target}"
    elif target.startswith(":"):
      target = f"@{base_repo}//{base_package_name}{target}"
    else:
      target = f"@{base_repo}//{base_package_name}:{target}"
  elif repo_mapping:
    parsed = parse_label(target)
    mapped_repo = repo_mapping.get(parsed.repo_name)
    if mapped_repo is not None:
      target = f"@{mapped_repo}//{parsed.package_name}:{parsed.target_name}"

  return target
