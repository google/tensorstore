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

import asyncio
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


def test_promise_result_release_gil():
  promise, future = ts.Promise.new()
  t = threading.Thread(target=future.result)
  t.start()
  time.sleep(0.1)
  promise.set_result(5)
  t.join()


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
@pytest.mark.skipif(
    'signal.getsignal(signal.SIGINT) != signal.default_int_handler',
    reason='SIGINT handler not installed')
def test_promise_wait_interrupt():

  promise, future = ts.Promise.new()

  def do_interrupt():
    time.sleep(0.01)
    sig = signal.CTRL_C_EVENT if os.name == 'nt' else signal.SIGINT  # type: ignore
    os.kill(os.getpid(), sig)

  with pytest.raises(KeyboardInterrupt):
    threading.Thread(target=do_interrupt).start()
    future.result(timeout=5)


def test_promise_cancel():
  promise, future = ts.Promise.new()
  assert future.done() == False

  def do_cancel():
    time.sleep(0.1)
    future.cancel()

  t = threading.Thread(target=do_cancel)
  t.start()
  with pytest.raises(asyncio.CancelledError):
    future.result(timeout=5)
  t.join()


def test_promise_timeout():
  promise, future = ts.Promise.new()
  assert future.done() == False
  with pytest.raises(TimeoutError):
    future.result(timeout=0.1)
  with pytest.raises(TimeoutError):
    future.result(deadline=time.time() + 0.1)
  promise.set_result(5)
  assert future.result(timeout=0) == 5
