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
"""CMake Provider types.

These providers are the primary way in which bazel_to_cmake references
bazel build targets.
"""

from typing import Tuple, Union

from .cmake_target import CMakePackage
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetPair
from .starlark.provider import Provider


class CMakeHallucinatedTarget(Provider):
  """CMake target corresponding to an unknown Bazel target."""

  __slots__ = ("target",)

  def __init__(self, target: CMakeTarget):
    assert isinstance(target, CMakeTarget), f"{repr(target)}"
    self.target = target

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.target)})"


class CMakePackageDepsProvider(Provider):
  """CMake packages required by a Bazel target."""

  __slots__ = ("package",)

  def __init__(self, package: CMakePackage):
    assert isinstance(package, CMakePackage), f"{repr(package)}"
    self.package = package

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.package)})"


class CMakeAddDependenciesProvider(Provider):
  """CMake add_dependencies required by a Bazel target."""

  __slots__ = ("target",)

  def __init__(self, target: CMakeTarget):
    assert isinstance(target, CMakeTarget), f"{repr(target)}"
    self.target = target

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.target)})"


class CMakeLinkLibrariesProvider(Provider):
  """CMake link_libraries required by a Bazel target."""

  __slots__ = ("target",)

  def __init__(self, target: CMakeTarget):
    assert isinstance(target, CMakeTarget), f"{repr(target)}"
    self.target = target

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.target)})"


class CMakeExecutableTargetProvider(Provider):
  """CMake target corresponding to a Bazel executable target."""

  __slots__ = ("target",)

  def __init__(self, target: CMakeTarget):
    assert isinstance(target, CMakeTarget), f"{repr(target)}"
    self.target = target

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.target)})"


class CMakeAliasProvider(Provider):
  """Indicates that this target is aliased to another target."""

  __slots__ = ("target",)

  def __init__(self, target: CMakeTarget):
    assert isinstance(target, CMakeTarget), f"{repr(target)}"
    self.target = target

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.target)})"


def make_providers(
    source: Union[CMakeTarget, CMakeTargetPair], *providers
) -> Tuple[Provider, ...]:
  """Construct providers from a CMakeTarget or CMakeTargetPair."""
  result = ()
  if isinstance(source, CMakeTargetPair):
    if CMakePackageDepsProvider in providers:
      result += (CMakePackageDepsProvider(source.cmake_package),)
    source = source.dep

  if isinstance(source, CMakeTarget):
    if CMakeAddDependenciesProvider in providers:
      result += (CMakeAddDependenciesProvider(source),)
    if CMakeLinkLibrariesProvider in providers:
      result += (CMakeLinkLibrariesProvider(source),)
    if CMakeExecutableTargetProvider in providers:
      result += (CMakeExecutableTargetProvider(source),)

  assert result
  return result


def default_providers(source: CMakeTargetPair) -> Tuple[Provider, ...]:
  return make_providers(
      source,
      CMakePackageDepsProvider,
      CMakeLinkLibrariesProvider,
  )
