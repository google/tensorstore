# Copyright 2024 The TensorStore Authors
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
"""Provides a collector interface for aggregating dependencies.

Example:
  from .starlark.common_provider import FilesProvider
  from .provider_util import ProviderCollection

  srcs_collector = ProviderCollection(FilesProvider)
  state.collect_targets(resolved_srcs, srcs_collector)

  srcs = sorted(srcs_collector.file_paths())
"""

# pylint: disable=missing-function-docstring,relative-beyond-top-level,missing-class-docstring

from typing import Iterator, Type, TypeVar

from .cmake_provider import CMakeAddDependenciesProvider
from .cmake_provider import CMakeExecutableTargetProvider
from .cmake_provider import CMakeHallucinatedTarget
from .cmake_provider import CMakeLinkLibrariesProvider
from .cmake_target import CMakeTarget
from .starlark.bazel_target import TargetId
from .starlark.common_providers import FilesProvider
from .starlark.provider import Provider
from .starlark.provider import TargetInfo

P = TypeVar('P', bound=Provider)


class ProviderCollection:
  """Collects TargetInfo for multiple targets."""

  __slots__ = ('_providers', '_assertions')

  def __init__(self):
    self._providers = {}

  def __repr__(self):
    return f'{self.__class__.__name__}(providers={self._providers})'

  def collect(self, target_id: TargetId, target_info: TargetInfo):
    self._providers[target_id] = target_info

  def assert_provider(self, t: Type[P]):
    failed = []
    for target_id, target_info in self._providers.items():
      if target_info.get(t) is None:
        failed.append(target_id.as_label())
    if failed:
      raise ValueError(
          f'Failed to collect type {t.__name__} from Targets {" ".join(failed)}'
      )

  def items(self, t: Type[P]) -> Iterator[P]:
    for x in self._providers.values():
      provider = x.get(t)
      if provider is not None:
        yield provider

  def targets(self) -> Iterator[CMakeTarget]:
    seen = set()
    for t in (
        CMakeHallucinatedTarget,
        CMakeAddDependenciesProvider,
        CMakeLinkLibrariesProvider,
        CMakeExecutableTargetProvider,
    ):
      for provider in self.items(t):
        if provider.target not in seen:
          seen.add(provider.target)
          yield provider.target

  def file_paths(self) -> Iterator[str]:
    for files_provider in self.items(FilesProvider):
      yield from files_provider.paths

  def add_dependencies(self) -> Iterator[CMakeTarget]:
    for add_dependencies_provider in self.items(CMakeAddDependenciesProvider):
      yield add_dependencies_provider.target

  def link_libraries(self) -> Iterator[CMakeTarget]:
    for t in (CMakeHallucinatedTarget, CMakeLinkLibrariesProvider):
      for provider in self.items(t):
        yield provider.target
