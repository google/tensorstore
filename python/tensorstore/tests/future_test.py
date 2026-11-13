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


def test_future_concurrent_ops() -> None:
  """Test multi-threaded races across Future and Promise operations."""

  def _do_concurrent_ops(iteration: int) -> None:
    callback_called = threading.Event()
    callback_result: list[Any] = [None]
    callback_error: list[BaseException | None] = [None]
    waiter_outcomes: list[tuple[Any, BaseException | None]] = []
    exc_waiter_outcomes: list[tuple[Any, BaseException | None]] = []
    outcomes_lock = threading.Lock()
    expected_value = [42, {'iteration': iteration}]

    def callback(f: ts.Future) -> None:
      try:
        callback_result[0] = f.result()
      except BaseException as e:  # pylint: disable=broad-exception-caught
        callback_error[0] = e
      finally:
        callback_called.set()

    def removable_callback(f: ts.Future) -> None:
      del f

    promise, future = ts.Promise.new()
    stop = threading.Event()
    num_workers = 8
    barrier = threading.Barrier(num_workers)

    def do_add_callback() -> None:
      barrier.wait(timeout=5)
      future.add_done_callback(callback)

    def do_add_and_remove_callback() -> None:
      barrier.wait(timeout=5)
      future.add_done_callback(removable_callback)
      future.remove_done_callback(removable_callback)

    def do_cancel() -> None:
      barrier.wait(timeout=5)
      future.cancel()

    def do_set_result() -> None:
      barrier.wait(timeout=5)
      try:
        promise.set_result(expected_value)
      except BaseException:  # pylint: disable=broad-exception-caught
        pass

    def do_set_exception() -> None:
      barrier.wait(timeout=5)
      try:
        promise.set_exception(ValueError('concurrent error'))
      except BaseException:  # pylint: disable=broad-exception-caught
        pass

    def do_wait_result() -> None:
      barrier.wait(timeout=5)
      try:
        res = future.result(timeout=5)
        with outcomes_lock:
          waiter_outcomes.append((res, None))
      except BaseException as e:  # pylint: disable=broad-exception-caught
        with outcomes_lock:
          waiter_outcomes.append((None, e))

    def do_wait_exception() -> None:
      barrier.wait(timeout=5)
      try:
        exc = future.exception(timeout=5)
        with outcomes_lock:
          exc_waiter_outcomes.append((exc, None))
      except BaseException as e:  # pylint: disable=broad-exception-caught
        with outcomes_lock:
          exc_waiter_outcomes.append((None, e))

    def read_props() -> None:
      while not stop.is_set():
        _ = future.done()
        _ = future.cancelled()

    worker_threads = [
        threading.Thread(target=do_set_result),
        threading.Thread(target=do_set_exception),
        threading.Thread(target=do_add_callback),
        threading.Thread(target=do_add_and_remove_callback),
        threading.Thread(target=do_cancel),
        threading.Thread(target=do_cancel),
        threading.Thread(target=do_wait_result),
        threading.Thread(target=do_wait_exception),
    ]
    assert len(worker_threads) == num_workers
    random.shuffle(worker_threads)
    props_thread = threading.Thread(target=read_props)
    threads = worker_threads + [props_thread]

    for t in threads:
      t.start()

    for t in worker_threads:
      t.join(timeout=10)
      assert not t.is_alive()
    stop.set()
    props_thread.join(timeout=10)
    assert not props_thread.is_alive()

    assert future.done()
    assert callback_called.wait(timeout=5)

    if future.cancelled():
      assert isinstance(callback_error[0], asyncio.CancelledError)
      with pytest.raises(asyncio.CancelledError):
        future.result()
      with pytest.raises(asyncio.CancelledError):
        future.exception()
      for res, err in waiter_outcomes:
        assert res is None
        assert isinstance(err, asyncio.CancelledError)
      for exc, err in exc_waiter_outcomes:
        assert exc is None
        assert isinstance(err, asyncio.CancelledError)
    elif future.exception() is not None:
      assert isinstance(future.exception(), ValueError)
      assert isinstance(callback_error[0], ValueError)
      with pytest.raises(ValueError, match='concurrent error'):
        future.result()
      for res, err in waiter_outcomes:
        assert res is None
        assert isinstance(err, ValueError)
      for exc, err in exc_waiter_outcomes:
        assert err is None
        assert isinstance(exc, ValueError)
    else:
      assert callback_error[0] is None
      assert future.result() == expected_value
      assert callback_result[0] == expected_value
      for res, err in waiter_outcomes:
        assert err is None
        assert res == expected_value
      for exc, err in exc_waiter_outcomes:
        assert err is None
        assert exc is None

  for i in range(100):
    _do_concurrent_ops(i)


def test_future_concurrent_dealloc() -> None:
  """Test races between Promise fulfillment and Future/Promise deallocation."""

  def _do_dealloc_race(iteration: int) -> None:
    pair: list[Any] = list(ts.Promise.new())
    barrier = threading.Barrier(2)

    def do_resolve() -> None:
      promise = pair[0]
      pair[0] = None
      barrier.wait(timeout=5)
      if iteration % 3 == 0:
        promise.set_result([iteration, {'k': iteration}])
      elif iteration % 3 == 1:
        promise.set_exception(ValueError(f'err {iteration}'))
      del promise

    def do_drop_future() -> None:
      future = pair[1]
      pair[1] = None
      barrier.wait(timeout=5)
      if iteration % 2 == 0:
        future.cancel()
      del future

    t1 = threading.Thread(target=do_resolve)
    t2 = threading.Thread(target=do_drop_future)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not t1.is_alive()
    assert not t2.is_alive()

  for i in range(100):
    _do_dealloc_race(i)
