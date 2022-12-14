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
from typing import NewType, List, Optional, Tuple, Union

from .starlark.bazel_target import TargetId
from .starlark.provider import Provider

_SPLIT_RE = re.compile("[:/]+")

CMakePairTupleType = Tuple[Optional[str], str, Optional[str]]
CMakeTarget = NewType("CMakeTarget", str)


class CMakeTargetPair(tuple):
  """CMakeTargetPair identifies a cmake target, optionally with an alias."""
  __slots__ = ()
  __str__ = None

  def __new__(cls,
              cmake_package: Union[Optional[str], CMakePairTupleType],
              target: Optional[CMakeTarget] = None,
              alias: Optional[CMakeTarget] = None):
    if target is not None:
      return tuple.__new__(cls, (cmake_package, target, alias))
    elif isinstance(
        cmake_package,
        tuple) and len(cmake_package) == 3 and cmake_package[1] is not None:
      return tuple.__new__(cls, cmake_package)
    else:
      raise ValueError(
          f"""CMakeTargetPair.__new__({cmake_package},{target},{alias})""")

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}({repr(self.cmake_package)},{repr(self.target)},{repr(self.alias)})"

  def with_alias(self, alias: Optional[CMakeTarget]) -> "CMakeTargetPair":
    return CMakeTargetPair(self.cmake_package, self.target, alias)

  @property
  def cmake_package(self) -> Optional[str]:
    return self.__getitem__(0)

  @property
  def target(self) -> CMakeTarget:
    return self.__getitem__(1)

  @property
  def alias(self) -> Optional[CMakeTarget]:
    return self.__getitem__(2)

  @property
  def dep(self) -> CMakeTarget:
    return self.alias or self.target

  def as_providers(self):
    return (CMakeTargetPairProvider(self), CMakeTargetProvider(self.target),
            CMakeDepsProvider([self.dep]))


def label_to_generated_cmake_target(target_id: TargetId,
                                    cmake_project: str) -> CMakeTargetPair:
  """Computes the generated CMake target corresponding to a Bazel target."""

  parts: List[str] = []
  parts.extend(x for x in _SPLIT_RE.split(target_id.package_name) if x)
  parts.extend(x for x in _SPLIT_RE.split(target_id.target_name) if x)
  if parts[0].lower() == cmake_project.lower() and len(parts) > 1:
    parts = parts[1:]

  if len(parts) >= 2 and parts[-1] == parts[-2]:
    parts = parts[:-1]
  target_name = "_".join(parts)

  return CMakeTargetPair(cmake_project,
                         CMakeTarget(f"{cmake_project}_{target_name}"),
                         CMakeTarget(f"{cmake_project}::{target_name}"))


class CMakeTargetPairProvider(Provider):
  """CMakeTargetPair provider decribing a target."""

  __slots__ = ("target_pair",)

  def __init__(self, target_pair: CMakeTargetPair):
    self.target_pair = target_pair

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.target_pair)})"

  @property
  def dep(self) -> CMakeTarget:
    return self.target_pair.dep

  @property
  def target(self) -> CMakeTarget:
    return self.target_pair.target


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
