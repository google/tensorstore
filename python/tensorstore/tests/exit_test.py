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

# Tests that TensorStore doesn't cause crashes when invoked while the
# interpreter is exiting.
#
# When run with pytest as the only test target, pytest finds no tests, since
# none are defined here, and calls `sys.exit(5)`.  If `run_with_finalization` is
# successful, it overrides the exit code by calling `os._exit`.

import asyncio
import atexit
import os
import sys
import traceback

import tensorstore as ts


def run_during_finalization():
  """Test function called when `sys.is_finalizing == True`."""
  sys.stdout.write('Running finalization test\n')
  sys.stdout.flush()
  try:

    async def test_read():
      t = ts.array([1, 2, 3], dtype=ts.int64)
      await asyncio.wait_for(t.read(), timeout=1)
      # Normally, await won't suceed.  However, await may still succeed, if it
      # happens that the read completed before the call to `await`.
      os._exit(0)

    asyncio.run(test_read())
  except asyncio.TimeoutError:
    # Expected behavior
    os._exit(0)
  except:  # pylint: disable=bare-except
    sys.stderr.write('Unexpected error\n')
    sys.stderr.flush()
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(3)


def on_exit():
  """Ensures `run_during_finalization` is called while the interpreter is exiting.

  Python does not provide a direct mechanism for running a function when the
  interpreter is exiting: `atxit` functions run before finalization.  However,
  Python does call `flush` on `sys.stdout` and `sys.stderr` during finalization.
  Therefore, we monkey patch `sys.stderr.flush` to call
  `run_during_finalization`.
  """
  orig_flush = sys.stderr.flush

  def do_flush(*args, **kwargs):
    orig_flush(*args, **kwargs)
    if not sys.is_finalizing:
      return
    sys.stderr.flush = orig_flush
    run_during_finalization()

  sys.stderr.flush = do_flush


atexit.register(on_exit)
