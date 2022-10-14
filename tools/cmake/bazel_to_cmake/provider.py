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
"""Defines information about analyzed Bazel targets."""

# pylint: disable=relative-beyond-top-level

from typing import Any, TypeVar, Type, Optional, List, NamedTuple, cast

from .label import CMakeTarget


class Provider:
  __slots__ = ()


class BuildSettingProvider(Provider):
  """Build setting value (i.e. flag value) corresponding to a Bazel target."""
  __slots__ = ("value",)

  def __init__(self, value: Any):
    self.value = value


class ConditionProvider(Provider):
  """Condition value corresponding to a Bazel target."""
  __slots__ = ("value",)

  def __init__(self, value: bool):
    self.value = value


class FilesProvider(Provider):
  """Files corresponding to a Bazel target."""

  __slots__ = ("paths",)

  def __init__(self, paths: List[str]):
    self.paths = paths


class CMakeTargetPair(NamedTuple):
  target: CMakeTarget
  alias: Optional[CMakeTarget]

  @property
  def dep(self):
    return self.alias or self.target

  def as_providers(self):
    return (CMakeTargetProvider(self.target), CMakeDepsProvider([self.dep]))


class CMakeDepsProvider(Provider):
  """CMake deps corresponding to a Bazel target."""

  __slots__ = ("targets",)

  def __init__(self, targets: List[CMakeTarget]):
    self.targets = targets


class CMakePackageDepsProvider(Provider):
  """CMake packages required by a Bazel target."""

  __slots__ = ("packages",)

  def __init__(self, packages: List[str]):
    self.packages = packages


class CMakeTargetProvider(Provider):
  """CMake target corresponding to a Bazel target."""

  __slots__ = ("target",)

  def __init__(self, target: CMakeTarget):
    self.target = target


P = TypeVar("P", bound=Provider)


class TargetInfo:
  """Providers associated with an analyzed Bazel target."""
  __slots__ = ("_providers",)

  def __init__(self, *args: Provider):
    providers = {}
    for provider in args:
      providers[type(provider)] = provider
    self._providers = providers

  def __getitem__(self, provider_type: Type[P]) -> P:
    return cast(P, self._providers[provider_type])

  def get(self, provider_type: Type[P]) -> Optional[P]:
    return cast(Optional[P], self._providers.get(provider_type))

  def __repr__(self):
    return ("{" + ", ".join(repr(value) for value in self._providers.values()) +
            "}")
