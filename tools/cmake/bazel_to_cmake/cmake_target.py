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

from typing import List, NamedTuple, NewType, Optional, Type, TypeVar

from .starlark.provider import Provider

CMakePackage = NewType("CMakePackage", str)
CMakeTarget = NewType("CMakeTarget", str)


class CMakePackageDepsProvider(Provider):
  """CMake packages required by a Bazel target."""

  __slots__ = ("packages",)

  def __init__(self, packages: List[CMakePackage]):
    self.packages = packages

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.packages)})"


class CMakeDepsProvider(Provider):
  """CMake deps corresponding to a Bazel target."""

  __slots__ = ("targets",)

  def __init__(self, targets: List[CMakeTarget]):
    self.targets = targets

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.targets)})"


class CMakeLibraryTargetProvider(Provider):
  """CMake target corresponding to a Bazel library target."""

  __slots__ = ("target",)

  def __init__(self, target: CMakeTarget):
    self.target = target

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.target)})"


class CMakeExecutableTargetProvider(Provider):
  """CMake target corresponding to a Bazel executable target."""

  __slots__ = ("target",)

  def __init__(self, target: CMakeTarget):
    self.target = target

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.target)})"


AnyCMakeTargetProvider = TypeVar(
    "AnyCMakeTargetProvider",
    CMakeLibraryTargetProvider,
    CMakeExecutableTargetProvider,
)


class CMakeTargetPair(NamedTuple):
  """CMakeTargetPair identifies a cmake target, optionally with an alias."""

  cmake_package: Optional[CMakePackage]
  target: CMakeTarget
  alias: Optional[CMakeTarget] = None

  def with_alias(self, alias: Optional[CMakeTarget]) -> "CMakeTargetPair":
    return self._replace(alias=alias)

  @property
  def dep(self) -> CMakeTarget:
    return self.alias or self.target

  def as_providers(
      self,
      provider: Optional[
          Type[AnyCMakeTargetProvider]
      ] = CMakeLibraryTargetProvider,
  ):
    a = (provider(self.target),) if provider is not None else tuple()
    return (
        CMakeDepsProvider([self.dep]),
        CMakePackageDepsProvider([self.cmake_package]),
    ) + a

  def __str__(self) -> str:
    raise NotImplementedError
