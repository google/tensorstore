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

import os
import re

from .bazel_glob import glob
from .bazel_glob import glob_pattern_to_regexp


def test_basic(tmpdir):
  paths = set([
      'BUILD',
      'a/a1.cc',
      'a/a2.cc',
      'a/b/BUILD',
      'a/b/ab1.cc',
      'a/b/c/abc.cc',
      'a/c/ac.cc',
  ])

  directory = str(tmpdir)
  os.chdir(directory)
  for filename in paths:
    d = os.path.dirname(filename)
    if d:
      os.makedirs(d, exist_ok=True)
    with open(filename, 'w') as f:
      f.write('')

  assert glob(directory, ['**/*.cc']) == sorted([
      'a/a1.cc',
      'a/a2.cc',
      'a/c/ac.cc',
  ])


def test_glob_exclude_matches_top_level_files(tmpdir):
  paths = set([
      'a.cc',
      'a_test.cc',
      'sub/b.cc',
      'sub/b_test.cc',
      'sub/deep/c.cc',
  ])

  directory = str(tmpdir)
  for filename in paths:
    d = os.path.dirname(filename)
    if d:
      os.makedirs(os.path.join(directory, d), exist_ok=True)
    with open(os.path.join(directory, filename), 'w') as f:
      f.write('')

  # `**/` at the start of an exclusion pattern matches zero or more
  # directories, so a top-level file must be excluded as well.
  assert glob(
      directory, ['**/*.cc'], exclude=['**/*_test.cc']
  ) == sorted(['a.cc', 'sub/b.cc', 'sub/deep/c.cc'])


def test_glob_pattern_to_regexp_leading_doublestar():
  # A pattern like `**/*_test.cc` matches files both in subdirectories and at
  # the top level.
  r = re.compile(glob_pattern_to_regexp('**/*_test.cc'))
  assert r.fullmatch('a_test.cc')
  assert r.fullmatch('sub/x_test.cc')
  assert r.fullmatch('sub/deep/x_test.cc')
  assert not r.fullmatch('sub/a.cc')

  r = re.compile(glob_pattern_to_regexp('**/test/**'))
  assert r.fullmatch('test/foo.cc')
  assert r.fullmatch('a/test/foo.cc')
  assert r.fullmatch('a/b/test/foo.cc')
  assert not r.fullmatch('a/foo.cc')
