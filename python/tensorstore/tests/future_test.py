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
import pickle
import random
import signal
import threading
import time
import types
from typing import Any, Callable, assert_type

import pytest
import tensorstore as ts

GcTester = Callable[[Any], None]


def test_promise_new() -> None:
  promise, future = ts.Promise.new()
  assert not future.done()
  promise.set_result(5)
  assert future.done()
  assert future.result() == 5


def test_promise_new_typed() -> None:
  promise, future = ts.Promise[int].new()
  assert_type(promise, ts.Promise[int])  # pytype: disable=assert-type
  assert_type(future, ts.Future[int])  # pytype: disable=assert-type
  assert not future.done()
  promise.set_result(5)
  assert future.done()
  assert future.result() == 5
  assert_type(future.result(), int)  # pytype: disable=assert-type


def test_type_params() -> None:
  assert ts.Promise[int] == types.GenericAlias(ts.Promise, (int,))
  assert ts.Future[int] == types.GenericAlias(ts.Future, (int,))


def test_promise_result_release_gil() -> None:
  promise, future = ts.Promise.new()
  t = threading.Thread(target=future.result)
  t.start()
  time.sleep(0.1)
  promise.set_result(5)
  t.join()


def test_promise_set_exception() -> None:
  promise, future = ts.Promise.new()
  assert not future.done()
  promise.set_exception(ValueError('abc'))
  with pytest.raises(ValueError, match='abc'):
    future.result()
  assert isinstance(future.exception(), ValueError)


@pytest.mark.skipif(
    os.name == 'nt',
    reason=(
        'CTRL_C_EVENT is delayed on Windows until keyboard input is received'
    ),
)
@pytest.mark.skipif(
    'signal.getsignal(signal.SIGINT) != signal.default_int_handler',
    reason='SIGINT handler not installed',
)
def test_promise_wait_interrupt() -> None:
  promise, future = ts.Promise.new()

  def do_interrupt() -> None:
    time.sleep(0.01)
    sig = signal.CTRL_C_EVENT if os.name == 'nt' else signal.SIGINT  # type: ignore
    os.kill(os.getpid(), sig)

  with pytest.raises(KeyboardInterrupt):
    threading.Thread(target=do_interrupt).start()
    future.result(timeout=5)


def test_promise_cancel() -> None:
  promise, future = ts.Promise.new()
  assert not future.done()

  def do_cancel() -> None:
    time.sleep(0.1)
    future.cancel()

  t = threading.Thread(target=do_cancel)
  t.start()
  with pytest.raises(asyncio.CancelledError):
    future.result(timeout=5)
  t.join()


def test_promise_timeout() -> None:
  promise, future = ts.Promise.new()
  assert not future.done()
  with pytest.raises(TimeoutError):
    future.result(timeout=0.1)
  with pytest.raises(TimeoutError):
    future.result(deadline=time.time() + 0.1)
  promise.set_result(5)
  assert future.result(timeout=0) == 5


async def test_coroutine() -> None:
  async def do_async() -> int:
    return 42

  assert await ts.Future(do_async()) == 42


async def test_coroutine_explicit_loop() -> None:
  data = threading.local()

  loop_promise, loop_future = ts.Promise.new()

  loop: asyncio.AbstractEventLoop

  def thread_proc() -> None:
    nonlocal loop
    data.thread = 'new'
    loop = asyncio.new_event_loop()
    loop_promise.set_result(loop)
    loop.run_forever()

  t = threading.Thread(target=thread_proc)
  t.start()

  loop = await loop_future

  async def do_async() -> str:
    return data.thread

  data.thread = 'main'

  assert await ts.Future(do_async()) == 'main'
  assert await ts.Future(do_async(), loop=loop) == 'new'

  loop.call_soon_threadsafe(loop.stop)

  t.join()


@pytest.mark.filterwarnings(
    'ignore:coroutine .* was never awaited:RuntimeWarning'
)
def test_coroutine_no_event_loop_specified() -> None:
  async def do_async() -> int:
    return 42

  with pytest.raises(ValueError, match='no event loop specified'):
    ts.Future(do_async())


def test_gc_result_cycle(gc_tester: GcTester) -> None:
  obj: list[ts.Future] = []
  f = ts.Future(obj)
  obj.append(f)
  gc_tester(f)


def test_gc_callback_cycle(gc_tester: GcTester) -> None:
  def callback(f: ts.Future) -> None:
    del f

  promise, future = ts.Promise.new()
  future.add_done_callback(callback)
  callback.future = future  # type: ignore
  callback.promise = promise  # type: ignore

  gc_tester(future)


def test_promise_instantiation() -> None:
  with pytest.raises(TypeError):
    ts.Promise()


def test_write_futures_instantiation() -> None:
  with pytest.raises(TypeError):
    ts.WriteFutures()


def test_pickle_failure() -> None:
  p, f = ts.Promise.new()
  with pytest.raises(TypeError):
    pickle.dumps(p)
  with pytest.raises(TypeError):
    pickle.dumps(f)


def test_release_future_after_calling_add_done_callback() -> None:
  p, f = ts.Promise.new()
  result = None

  def callback(f: ts.Future) -> None:
    nonlocal result
    result = f.result()

  f.add_done_callback(callback)
  del f
  p.set_result(10)
  assert result == 10


def test_promise_not_fulfilled() -> None:
  p, f = ts.Promise.new()

  exc = None

  def callback(f: ts.Future) -> None:
    nonlocal exc
    exc = f.exception()

  f.add_done_callback(callback)

  del p
  assert f.done()
  assert exc is not None


def test_future_concurrent_wait_cancel() -> None:
  _, future = ts.Promise.new()

  def do_wait() -> None:
    with pytest.raises(asyncio.CancelledError):
      future.result(timeout=5)

  threads = [threading.Thread(target=do_wait) for _ in range(4)]
  for t in threads:
    t.start()

  time.sleep(0.01)
  future.cancel()

  for t in threads:
    t.join()


def test_future_concurrent_wait() -> None:
  promise, future = ts.Promise.new()

  def do_wait() -> None:
    with pytest.raises(TimeoutError):
      future.result(timeout=0.01)
    assert future.result(timeout=5) == 5

  threads = [threading.Thread(target=do_wait) for _ in range(4)]
  for t in threads:
    t.start()

  time.sleep(0.1)
  promise.set_result(5)

  for t in threads:
    t.join()


def _get_delay() -> float:
  return random.random() * 0.01


def _run_threads(
    promise: ts.Promise,
    i: int,
    stop: threading.Event,
    threads: list[threading.Thread],
) -> None:
  """Runs a list of threads concurrently with setting the promise result."""

  delay = _get_delay()
  for t in threads:
    t.start()

  time.sleep(delay)
  try:
    promise.set_result(i)
  except:  # pylint: disable=bare-except
    pass

  time.sleep(0.01)
  stop.set()

  for t in threads:
    t.join()


def test_future_concurrent_set_cancel_callback() -> None:
  """Test multi-treaded races between adding callbacks and cancellation."""

  def _concurrent_set_cancel_callback(i: int) -> None:
    callback_called = threading.Event()
    callback_result = [None]
    callback_error = [None]

    def callback(f: ts.Future) -> None:
      nonlocal callback_called
      nonlocal callback_result
      nonlocal callback_error
      try:
        callback_result[0] = f.result()
      except asyncio.CancelledError as e:
        callback_error[0] = e
      except Exception as e:  # pylint: disable=broad-exception-caught
        callback_error[0] = e
      finally:
        callback_called.set()

    def do_add_callback(delay: float, f: ts.Future) -> None:
      time.sleep(delay)
      f.add_done_callback(callback)

    def do_cancel(delay: float, f: ts.Future) -> None:
      time.sleep(delay)
      f.cancel()

    promise, future = ts.Promise.new()
    _run_threads(
        promise,
        i,
        threading.Event(),  # unused
        [
            threading.Thread(target=do_cancel, args=(_get_delay(), future)),
            threading.Thread(
                target=do_add_callback, args=(_get_delay(), future)
            ),
        ],
    )

    assert future.done()
    assert callback_called.wait(timeout=5)

    if future.cancelled():
      assert isinstance(callback_error[0], asyncio.CancelledError)
      with pytest.raises(asyncio.CancelledError):
        future.result()
    else:
      # If not cancelled, result must have been set.
      assert callback_error[0] is None
      assert future.result() == i
      assert callback_result[0] == i

  for i in range(20):
    _concurrent_set_cancel_callback(i)


def test_future_concurrent_ops() -> None:
  """Test multi-treaded races between adding callbacks and setting result."""

  def _concurrent_ops() -> None:
    events = [threading.Event() for _ in range(8)]
    results = {}
    lock = threading.Lock()
    stop = threading.Event()

    def make_callback(
        idx: int, event: threading.Event
    ) -> Callable[[ts.Future], None]:
      def callback(f: ts.Future) -> None:
        try:
          result = f.result()
          with lock:
            results[idx] = result
        finally:
          event.set()

      return callback

    promise, future = ts.Promise.new()

    def do_add_callback(
        delay: float, callback: Callable[[ts.Future], None]
    ) -> None:
      time.sleep(delay)
      future.add_done_callback(callback)

    def read_props() -> None:
      while not stop.is_set():
        _ = future.done()
        _ = future.cancelled()

    threads = []
    for i in range(len(events)):
      threads.append(
          threading.Thread(
              target=do_add_callback,
              args=(_get_delay(), make_callback(i, events[i])),
          )
      )
      threads.append(threading.Thread(target=read_props))

    _run_threads(promise, 42, stop, threads)

    for i in range(len(events)):
      assert events[i].wait(timeout=5)
      assert results[i] == 42

  for _ in range(20):
    _concurrent_ops()
