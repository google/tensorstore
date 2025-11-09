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

from .bazel_glob import glob


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
