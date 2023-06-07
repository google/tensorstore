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

from typing import List, NamedTuple, NewType, Optional

from .starlark.provider import Provider

CMakePackage = NewType("CMakePackage", str)
CMakeTarget = NewType("CMakeTarget", str)


class CMakeTargetPair(NamedTuple):
  """CMakeTarget identifies a cmake target, optionally with an alias."""

  cmake_package: Optional[CMakePackage]
  target: CMakeTarget
  alias: Optional[CMakeTarget] = None

  def with_alias(self, alias: Optional[CMakeTarget]) -> "CMakeTargetPair":
    return self._replace(alias=alias)

  @property
  def dep(self) -> CMakeTarget:
    return self.alias or self.target

  def as_providers(self):
    return (
        CMakeTargetPairProvider(self),
        CMakeTargetProvider(self.target),
        CMakeDepsProvider([self.dep]),
    )

  def __str__(self) -> str:
    raise NotImplementedError


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

  def __init__(self, packages: List[CMakePackage]):
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
