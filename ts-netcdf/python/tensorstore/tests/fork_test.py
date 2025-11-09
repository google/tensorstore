# Copyright 2025 The TensorStore Authors
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

# Tests that TensorStore doesn't cause crashes when invoked while the
# interpreter is exiting.
#
# When run with pytest as the only test target, pytest finds no tests, since
# none are defined here, and calls `sys.exit(5)`.  If `run_with_finalization` is
# successful, it overrides the exit code by calling `os._exit`.


import os
import sys
import tempfile

import tensorstore as ts


def test_fork_without_tensorstore_use_does_not_crash():
  if sys.platform not in ('linux', 'darwin'):
    return
  with tempfile.TemporaryDirectory() as dir_path:
    spec = {
        'driver': 'file',
        'path': dir_path,
    }
    kv = ts.KvStore.open(spec).result()
    kv.list().result()
    pid = os.fork()
    if pid == 0:
      # Child does not use tensorstore
      os._exit(0)
    else:
      # Parent can still use tensorstore.
      kv.list().result()
      child = os.waitpid(pid, 0)
      assert child == (pid, 0)


def test_fork_with_tensorstore_use_does_crash():
  if sys.platform not in ('linux', 'darwin'):
    return
  with tempfile.TemporaryDirectory() as dir_path:
    spec = {
        'driver': 'file',
        'path': dir_path,
    }
    kv = ts.KvStore.open(spec).result()
    kv.list().result()
    pid = os.fork()
    if pid == 0:
      # Child can't use tensorstore.
      kv.list().result()
      os._exit(0)
    else:
      child = os.waitpid(pid, 0)
      assert child != (pid, 0)
