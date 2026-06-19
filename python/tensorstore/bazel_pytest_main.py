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

# This file is a template used by the implementation of
# `tensorstore_pytest_test`.  PYTEST_ARGS, PYTEST_TARGETS, and USE_ABSL will be
# substituted with constants.

import os
import stat
import sys

import pytest


def invoke_tests(argv):
  # Remove directory containing this script from the search path to avoid import
  # problems.
  del sys.path[0]
  implicit_args = PYTEST_ARGS
  if 'TEST_TMPDIR' in os.environ:  # running as bazel test
    # set pytest's cache dir to a location that it can safely write to
    implicit_args += [
        '--override-ini',
        'cache_dir=' + os.environ['TEST_TMPDIR'],
    ]
  if 'XML_OUTPUT_FILE' in os.environ:
    # Output per-test-case information in XML format to allow Bazel to aggregate
    # with greater detail.
    implicit_args += ['--junitxml', os.environ['XML_OUTPUT_FILE']]
  implicit_args += PYTEST_TARGETS
  print(implicit_args + argv[1:])

  # Monkey patch os.path.samefile and os.lstat only on Windows to avoid flakes
  # from the 'local-spawn-runner' race condition.
  #
  # Because Bazel on Windows lacks native mount namespaces for sandboxing,
  # concurrent Bazel test runs execute in a shared directory hierarchy.
  # During test discovery, Pytest lists files (finding temporary files named
  # 'local-spawn-runner.XXXX' files from peer actions) and subsequently queries
  # their metadata. If a concurrent Bazel action deletes these temporary files
  # after listing but before Pytest queries them, os.lstat or os.path.samefile
  # fails with a FileNotFoundError, causing the test runner execution to
  # crash/flake.
  if sys.platform == 'win32':
    orig_samefile = os.path.samefile

    def samefile(*args, **kwargs):
      try:
        return orig_samefile(*args, **kwargs)
      except FileNotFoundError:
        return False

    os.path.samefile = samefile

    orig_lstat = os.lstat

    def lstat(path, *args, **kwargs):
      try:
        return orig_lstat(path, *args, **kwargs)
      except FileNotFoundError:
        if 'local-spawn-runner' in str(path):
          # Return a stat result of a fake empty regular file.
          return os.stat_result((stat.S_IFREG, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        raise

    os.lstat = lstat

  return pytest.main(args=implicit_args + argv[1:])


def main():
  if USE_ABSL:
    import absl.app
    import absl.flags

    absl.flags.FLAGS.set_default('logtostderr', True)
    sys.argv.insert(1, '--')
    absl.app.run(invoke_tests)
  else:
    sys.exit(invoke_tests(sys.argv))


if __name__ == '__main__':
  main()
