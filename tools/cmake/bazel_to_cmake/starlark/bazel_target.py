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
"""Classes used as keys to identify bazel repositories, packages, and targets.

RepositoryId: Identify a repository
PackageId: Identify a package in a repository.
TargetId: Identify a target in a package and repository.
"""

# pylint: disable=missing-function-docstring,relative-beyond-top-level

import builtins
import operator
import os
from typing import Dict, NamedTuple, Optional, Union, Tuple, cast


class RepositoryId(NamedTuple):
  """RepositoryId identifies a repository."""
  repository_name: str

  def __repr__(self) -> str:
    return f'RepositoryId("{self.repository_name}")'

  def get_package_id(self, package_name: str) -> 'PackageId':
    """Returns the package in this repository."""
    return PackageId(
        repository_name=self.repository_name, package_name=package_name)

  def parse_target(self, target: str) -> 'TargetId':
    if target.startswith('@'):
      return parse_absolute_target(target)

    if not target.startswith('//'):
      raise ValueError(f"Invalid repository-relative label: {target} (no '//')")

    package = target[2:]
    if package.startswith('/'):
      raise ValueError(
          f"Invalid repository-relative label: {target} (starts with '/')")

    i = package.find(':')
    if i >= 0:
      return TargetId(
          repository_name=self.repository_name,
          package_name=package[:i],
          target_name=package[i + 1:])

    return TargetId(
        repository_name=self.repository_name,
        package_name=package,
        target_name=os.path.basename(package))


class PackageId(tuple):
  """PackageId identifies a package in a repository."""
  __slots__ = ()

  repository_name = builtins.property(operator.itemgetter(0))
  package_name = builtins.property(operator.itemgetter(1))

  def __new__(cls,
              repository_name: Union[str, Tuple[str, str]],
              package_name: Optional[str] = None):
    if None not in [repository_name, package_name]:
      return tuple.__new__(cls, (repository_name, package_name))
    elif isinstance(repository_name, tuple) and len(repository_name) == 2:
      return tuple.__new__(cls, repository_name)
    else:
      return parse_absolute_target(cast(str, repository_name)).package_id

  def __repr__(self) -> str:
    return f'PackageId("@{self.repository_name}//{self.package_name}")'

  @property
  def repository_id(self) -> RepositoryId:
    return RepositoryId(self.repository_name)

  def get_target_id(self, target_name: str) -> 'TargetId':
    return TargetId(
        repository_name=self.repository_name,
        package_name=self.package_name,
        target_name=target_name)

  def parse_target(self, target: str) -> 'TargetId':
    if not target:
      raise ValueError(f'Invalid empty label in package: {self.package_name}')
    if target.startswith('@'):
      return parse_absolute_target(target)
    if target.startswith('/'):
      return self.repository_id.parse_target(target)
    if target.startswith(':'):
      return TargetId(
          repository_name=self.repository_name,
          package_name=self.package_name,
          target_name=target[1:])
    return TargetId(
        repository_name=self.repository_name,
        package_name=self.package_name,
        target_name=target)


class TargetId(tuple):
  """TargetId identifies a target in a package."""
  __slots__ = ()

  repository_name = builtins.property(operator.itemgetter(0))
  package_name = builtins.property(operator.itemgetter(1))
  target_name = builtins.property(operator.itemgetter(2))

  def __new__(cls,
              repository_name: Union[str, Tuple[str, str, str]],
              package_name: Optional[str] = None,
              target_name: Optional[str] = None):
    if None not in [repository_name, package_name, target_name]:
      return tuple.__new__(cls, (repository_name, package_name, target_name))
    elif isinstance(repository_name, tuple) and len(repository_name) == 3:
      return tuple.__new__(cls, repository_name)
    elif package_name is not None or target_name is not None:
      raise ValueError('PackageId cannot be constructed')
    else:
      return parse_absolute_target(cast(str, repository_name))

  def __repr__(self) -> str:
    return f'TargetId({self.as_label()})'

  def as_label(self) -> str:
    return f'@{self.repository_name}//{self.package_name}:{self.target_name}'

  @property
  def repository_id(self) -> RepositoryId:
    """The RepositoryId which owns this target."""
    return RepositoryId(self.repository_name)

  @property
  def package_id(self) -> PackageId:
    """The PackageId which owns this target."""
    return PackageId(
        repository_name=self.repository_name, package_name=self.package_name)

  def get_target_id(self, target_name: str) -> 'TargetId':
    return TargetId(
        repository_name=self.repository_name,
        package_name=self.package_name,
        target_name=target_name)

  def parse_target(self, target: str) -> 'TargetId':
    return self.package_id.parse_target(target)


def parse_absolute_target(target: str) -> TargetId:
  """Parse an absolute label, in the form @repo/package:target."""
  if not target.startswith('@'):
    raise ValueError(f'Invalid absolute label: f{target} (no repo prefix)')

  i = target.find('//')
  if i < 0:
    # @repo => @repo//:repo
    return TargetId(
        repository_name=target[1:], package_name='', target_name=target[1:])

  # @repo//package:target
  return RepositoryId(target[1:i]).parse_target(target[i:])


def remap_target_repo(target: TargetId,
                      mapping: Optional[Dict[str, str]] = None) -> TargetId:
  """Apply a repository mapping to a TargetId."""
  if mapping is None:
    return target
  new_repository_name = mapping.get(target.repository_name)
  if new_repository_name is None:
    return target

  return RepositoryId(new_repository_name).get_package_id(
      target.package_name).get_target_id(target.target_name)
