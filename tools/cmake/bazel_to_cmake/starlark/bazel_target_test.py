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
import unittest

from .bazel_target import PackageId
from .bazel_target import parse_absolute_target
from .bazel_target import remap_target_repo
from .bazel_target import RepositoryId
from .bazel_target import TargetId


class TestBazelTarget(unittest.TestCase):

  def test_parse_absolute_target(self):
    self.assertEqual(
        TargetId(repository_name='foo', package_name='bar', target_name='bar'),
        parse_absolute_target('@foo//bar'))
    self.assertEqual(
        TargetId(repository_name='foo', package_name='', target_name='bar'),
        parse_absolute_target('@foo//:bar'))
    self.assertEqual(
        TargetId(repository_name='foo', package_name='bar', target_name='baz'),
        parse_absolute_target('@foo//bar:baz'))
    self.assertEqual(
        TargetId(repository_name='foo', package_name='', target_name='foo'),
        parse_absolute_target('@foo'))

    with self.assertRaises(Exception) as _:
      parse_absolute_target('')
    with self.assertRaises(Exception) as _:
      parse_absolute_target('@foo///bar')

  def test_parse_repository_relative_label(self):
    repo = RepositoryId('repo')
    self.assertEqual(
        PackageId(repository_name='repo', package_name='x'),
        repo.get_package_id('x'))

    # Absolute label
    self.assertEqual(
        TargetId(repository_name='foo', package_name='bar', target_name='bar'),
        repo.parse_target('@foo//bar'))
    self.assertEqual(
        TargetId(repository_name='foo', package_name='', target_name='bar'),
        repo.parse_target('@foo//:bar'))
    self.assertEqual(
        TargetId(repository_name='foo', package_name='', target_name='foo'),
        repo.parse_target('@foo'))

    # Relative label
    self.assertEqual(
        TargetId(repository_name='repo', package_name='bar', target_name='bar'),
        repo.parse_target('//bar'))
    self.assertEqual(
        TargetId(
            repository_name='repo', package_name='bar/baz', target_name='baz'),
        repo.parse_target('//bar/baz'))
    self.assertEqual(
        TargetId(repository_name='repo', package_name='bar', target_name='baz'),
        repo.parse_target('//bar:baz'))

    with self.assertRaises(Exception) as _:
      repo.parse_target('')
    with self.assertRaises(Exception) as _:
      repo.parse_target('///foo/bar')
    with self.assertRaises(Exception) as _:
      repo.parse_target('bar')

  def test_parse_package_relative_label(self):
    repo = RepositoryId('repo')

    # Absolute label
    self.assertEqual(
        TargetId(repository_name='foo', package_name='bar', target_name='bar'),
        repo.parse_target('@foo//bar'))
    self.assertEqual(
        TargetId(repository_name='foo', package_name='', target_name='bar'),
        repo.parse_target('@foo//:bar'))
    self.assertEqual(
        TargetId(repository_name='foo', package_name='', target_name='foo'),
        repo.parse_target('@foo'))

    # Relative label
    self.assertEqual(
        TargetId(repository_name='repo', package_name='bar', target_name='bar'),
        repo.parse_target('//bar'))
    self.assertEqual(
        TargetId(
            repository_name='repo', package_name='bar/baz', target_name='baz'),
        repo.parse_target('//bar/baz'))
    self.assertEqual(
        TargetId(repository_name='repo', package_name='bar', target_name='baz'),
        repo.parse_target('//bar:baz'))

    with self.assertRaises(Exception) as _:
      repo.parse_target('')
    with self.assertRaises(Exception) as _:
      repo.parse_target('///foo/bar')

  def test_remap_target(self):
    base = TargetId(
        repository_name='relative', package_name='package', target_name='name')

    self.assertEqual(base, remap_target_repo(base, {}))
    self.assertEqual(base, remap_target_repo(base, {'other': 'other'}))
    self.assertEqual(
        TargetId(
            repository_name='absolute',
            package_name='package',
            target_name='name'),
        remap_target_repo(base, {'relative': 'absolute'}))

  def test_pickle_unpickle(self):
    self.assertEqual(
        TargetId(('foo', 'bar', 'baz')),
        pickle.loads(pickle.dumps(TargetId(('foo', 'bar', 'baz')))))
    self.assertEqual(
        PackageId(('foo', 'bar')),
        pickle.loads(pickle.dumps(PackageId(('foo', 'bar')))))
    self.assertEqual(
        RepositoryId('foo'), pickle.loads(pickle.dumps(RepositoryId('foo'))))

  def test_str_constructors(self):
    self.assertEqual(TargetId(('foo', 'bar', 'baz')), TargetId('@foo//bar:baz'))
    self.assertEqual(PackageId(('foo', 'bar')), PackageId('@foo//bar'))
