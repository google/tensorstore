# Copyright 2020 The TensorStore Authors
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
"""Driver for running pytest from bazel."""

import os
import sys

import pytest


def main(argv):
  # Remove directory containing this script from the search path to avoid import
  # problems.
  del sys.path[0]
  args = argv[1:]
  if 'TEST_TMPDIR' in os.environ:  # running as bazel test
    # set pytest's cache dir to a location that it can safely write to
    args += ['--override-ini', 'cache_dir=' + os.environ['TEST_TMPDIR']]
  if 'XML_OUTPUT_FILE' in os.environ:
    # Output per-test-case information in XML format to allow Bazel to aggregate
    # with greater detail.
    args += ['--junitxml', os.environ['XML_OUTPUT_FILE']]
  return pytest.main(args=args)


if __name__ == '__main__':
  sys.exit(main(sys.argv))
