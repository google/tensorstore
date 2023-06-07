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

# pylint: disable=g-importing-member

import pickle

import pytest

from .bazel_target import apply_repo_mapping
from .bazel_target import PackageId
from .bazel_target import parse_absolute_target
from .bazel_target import RepositoryId
from .bazel_target import TargetId


def test_parse_absolute_target():
  assert TargetId(
      repository_name='foo', package_name='bar', target_name='bar'
  ) == parse_absolute_target('@foo//bar')
  assert TargetId(
      repository_name='foo', package_name='', target_name='bar'
  ) == parse_absolute_target('@foo//:bar')
  assert TargetId(
      repository_name='foo', package_name='bar', target_name='baz'
  ) == parse_absolute_target('@foo//bar:baz')
  assert TargetId(
      repository_name='foo', package_name='', target_name='foo'
  ) == parse_absolute_target('@foo')

  with pytest.raises(Exception):
    parse_absolute_target('')
  with pytest.raises(Exception):
    parse_absolute_target('@foo///bar')


def test_parse_repository_relative_label():
  repo = RepositoryId('repo')
  assert PackageId(
      repository_name='repo', package_name='x'
  ) == repo.get_package_id('x')

  # Absolute label
  assert TargetId(
      repository_name='foo', package_name='bar', target_name='bar'
  ) == repo.parse_target('@foo//bar')
  assert TargetId(
      repository_name='foo', package_name='', target_name='bar'
  ) == repo.parse_target('@foo//:bar')
  assert TargetId(
      repository_name='foo', package_name='', target_name='foo'
  ) == repo.parse_target('@foo')

  # Relative label
  assert TargetId(
      repository_name='repo', package_name='bar', target_name='bar'
  ) == repo.parse_target('//bar')
  assert TargetId(
      repository_name='repo', package_name='bar/baz', target_name='baz'
  ) == repo.parse_target('//bar/baz')
  assert TargetId(
      repository_name='repo', package_name='bar', target_name='baz'
  ) == repo.parse_target('//bar:baz')

  with pytest.raises(Exception):
    repo.parse_target('')
  with pytest.raises(Exception):
    repo.parse_target('///foo/bar')
  with pytest.raises(Exception):
    repo.parse_target('bar')


def test_parse_package_relative_label():
  repo = RepositoryId('repo')

  # Absolute label
  assert TargetId(
      repository_name='foo', package_name='bar', target_name='bar'
  ) == repo.parse_target('@foo//bar')
  assert TargetId(
      repository_name='foo', package_name='', target_name='bar'
  ) == repo.parse_target('@foo//:bar')
  assert TargetId(
      repository_name='foo', package_name='', target_name='foo'
  ) == repo.parse_target('@foo')

  # Relative label
  assert TargetId(
      repository_name='repo', package_name='bar', target_name='bar'
  ) == repo.parse_target('//bar')
  assert TargetId(
      repository_name='repo', package_name='bar/baz', target_name='baz'
  ) == repo.parse_target('//bar/baz')
  assert TargetId(
      repository_name='repo', package_name='bar', target_name='baz'
  ) == repo.parse_target('//bar:baz')

  with pytest.raises(Exception):
    repo.parse_target('')
  with pytest.raises(Exception):
    repo.parse_target('///foo/bar')


def test_remap_target():
  base = TargetId(
      repository_name='relative', package_name='package', target_name='name'
  )

  assert base == apply_repo_mapping(base, {})
  assert base == apply_repo_mapping(
      base, {RepositoryId('other'): RepositoryId('other')}
  )
  assert TargetId(
      repository_name='absolute', package_name='package', target_name='name'
  ) == apply_repo_mapping(
      base, {RepositoryId('relative'): RepositoryId('absolute')}
  )


def test_pickle_unpickle():
  assert TargetId('foo', 'bar', 'baz') == pickle.loads(
      pickle.dumps(TargetId('foo', 'bar', 'baz'))
  )
  assert PackageId('foo', 'bar') == pickle.loads(
      pickle.dumps(PackageId('foo', 'bar'))
  )
  assert RepositoryId('foo') == pickle.loads(pickle.dumps(RepositoryId('foo')))


def test_str_constructors():
  assert TargetId('foo', 'bar', 'baz') == TargetId.parse('@foo//bar:baz')
  assert PackageId('foo', 'bar') == PackageId.parse('@foo//bar')
