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
"""Tests for tensorstore.Transaction."""

import contextlib
import tempfile

import numpy as np
import pytest
import tensorstore as ts

pytestmark = pytest.mark.asyncio


@contextlib.contextmanager
def make_dataset():
  with tempfile.TemporaryDirectory() as dir_path:
    yield ts.open({
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': dir_path,
        },
        'metadata': {
            'compression': {
                'type': 'gzip'
            },
            'dataType': 'uint16',
            'dimensions': [3, 4],
            'blockSize': [2, 3],
        },
        'create': True,
        'delete_existing': True,
    }).result()


async def test_transaction_read_write():
  with make_dataset() as dataset:
    txn = ts.Transaction()
    assert not txn.aborted
    assert not txn.commit_started
    assert txn.open
    dataset.with_transaction(txn)[1:2, 3:4] = 42
    dataset.with_transaction(txn)[0:2, :1] = 5
    np.testing.assert_equal([
        [5, 0, 0, 0],
        [5, 0, 0, 42],
        [0, 0, 0, 0],
    ],
                            dataset.with_transaction(txn).read().result())
    np.testing.assert_equal([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
                            dataset.read().result())
    txn.commit_async().result()
    assert txn.commit_started
    assert not txn.open
    np.testing.assert_equal([
        [5, 0, 0, 0],
        [5, 0, 0, 42],
        [0, 0, 0, 0],
    ],
                            dataset.read().result())


async def test_transaction_context_manager_commit():
  with make_dataset() as dataset:
    with ts.Transaction() as txn:
      dataset.with_transaction(txn)[1:2, 3:4] = 42
      np.testing.assert_equal([
          [0, 0, 0, 0],
          [0, 0, 0, 42],
          [0, 0, 0, 0],
      ],
                              dataset.with_transaction(txn).read().result())
      np.testing.assert_equal([
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
      ],
                              dataset.read().result())
    np.testing.assert_equal([
        [0, 0, 0, 0],
        [0, 0, 0, 42],
        [0, 0, 0, 0],
    ],
                            dataset.read().result())


async def test_transaction_context_manager_abort():
  with make_dataset() as dataset:
    with pytest.raises(ValueError, match='want to abort'):
      with ts.Transaction() as txn:
        dataset.with_transaction(txn)[1:2, 3:4] = 42
        np.testing.assert_equal([
            [0, 0, 0, 0],
            [0, 0, 0, 42],
            [0, 0, 0, 0],
        ],
                                dataset.with_transaction(txn).read().result())
        raise ValueError('want to abort')
    np.testing.assert_equal([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
                            dataset.read().result())


async def test_transaction_async_context_manager_commit():
  with make_dataset() as dataset:
    async with ts.Transaction() as txn:
      dataset.with_transaction(txn)[1:2, 3:4] = 42
      np.testing.assert_equal([
          [0, 0, 0, 0],
          [0, 0, 0, 42],
          [0, 0, 0, 0],
      ],
                              dataset.with_transaction(txn).read().result())
      np.testing.assert_equal([
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
      ],
                              dataset.read().result())
    np.testing.assert_equal([
        [0, 0, 0, 0],
        [0, 0, 0, 42],
        [0, 0, 0, 0],
    ],
                            dataset.read().result())


async def test_transaction_async_context_manager_abort():
  with make_dataset() as dataset:
    with pytest.raises(ValueError, match='want to abort'):
      async with ts.Transaction() as txn:
        dataset.with_transaction(txn)[1:2, 3:4] = 42
        np.testing.assert_equal([
            [0, 0, 0, 0],
            [0, 0, 0, 42],
            [0, 0, 0, 0],
        ],
                                dataset.with_transaction(txn).read().result())
        raise ValueError('want to abort')
    np.testing.assert_equal([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
                            dataset.read().result())
