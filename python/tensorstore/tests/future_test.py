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

import os
import time
import signal
import threading

import pytest
import tensorstore as ts


def test_promise_new():

  promise, future = ts.Promise.new()
  assert future.done() == False
  promise.set_result(5)
  assert future.done()
  assert future.result() == 5


def test_promise_set_exception():

  promise, future = ts.Promise.new()
  assert future.done() == False
  promise.set_exception(RuntimeError(5))
  with pytest.raises(RuntimeError):
    future.result()
  assert isinstance(future.exception(), RuntimeError)


@pytest.mark.skipif(
    os.name == 'nt',
    reason='CTRL_C_EVENT is delayed on Windows until keyboard input is received'
)
def test_promise_wait_interrupt():

  promise, future = ts.Promise.new()
  event = threading.Event()

  def do_interrupt():
    time.sleep(0.001)
    sig = signal.CTRL_C_EVENT if os.name == 'nt' else signal.SIGINT
    os.kill(os.getpid(), sig)

  started = False
  value = None
  with pytest.raises(KeyboardInterrupt):
    threading.Thread(target=do_interrupt).start()
    value = future.result()
