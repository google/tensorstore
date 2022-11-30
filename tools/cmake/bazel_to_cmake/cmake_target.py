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
"""CMake Provider types."""

import re
from typing import NewType, List, NamedTuple, Optional

from .starlark.bazel_target import TargetId
from .starlark.provider import Provider

_SPLIT_RE = re.compile("[:/]+")

CMakeTarget = NewType("CMakeTarget", str)


def label_to_generated_cmake_target(target_id: TargetId,
                                    cmake_project: str,
                                    alias: bool = False) -> CMakeTarget:
  """Computes the generated CMake target corresponding to a Bazel target."""

  parts: List[str] = []
  parts.extend(x for x in _SPLIT_RE.split(target_id.package_name) if x)
  parts.extend(x for x in _SPLIT_RE.split(target_id.target_name) if x)
  if parts[0] == cmake_project and len(parts) > 1:
    parts = parts[1:]

  if len(parts) >= 2 and parts[-1] == parts[-2]:
    parts = parts[:-1]

  if alias:
    return CMakeTarget(cmake_project + "::" + "_".join(parts))
  return CMakeTarget(cmake_project + "_" + "_".join(parts))


class CMakeDepsProvider(Provider):
  """CMake deps corresponding to a Bazel target."""

  __slots__ = ("targets",)

  def __init__(self, targets: List[CMakeTarget]):
    self.targets = targets

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.targets)})"


class CMakePackageDepsProvider(Provider):
  """CMake packages required by a Bazel target."""

  __slots__ = ("packages",)

  def __init__(self, packages: List[str]):
    self.packages = packages

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.packages)})"


class CMakeTargetProvider(Provider):
  """CMake target corresponding to a Bazel target."""

  __slots__ = ("target",)

  def __init__(self, target: CMakeTarget):
    self.target = target

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.target)})"


class CMakeTargetPair(NamedTuple):
  target: CMakeTarget
  alias: Optional[CMakeTarget]

  @property
  def dep(self) -> CMakeTarget:
    return self.alias or self.target

  def as_providers(self):
    return (CMakeTargetProvider(self.target), CMakeDepsProvider([self.dep]))
