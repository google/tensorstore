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
"""Tests for tensorstore.KvStore."""

import copy
import pickle
import tempfile

import pytest
import tensorstore as ts


def test_instantiation():
  with pytest.raises(TypeError):
    ts.KvStore()


def test_spec_pickle():
  kv_spec = ts.KvStore.Spec('memory://')
  assert ts.KvStore.Spec.__module__ == 'tensorstore'
  assert ts.KvStore.Spec.__qualname__ == 'KvStore.Spec'
  assert pickle.loads(pickle.dumps(kv_spec)).to_json() == kv_spec.to_json()


def test_pickle():
  kv = ts.KvStore.open('memory://').result()
  assert pickle.loads(pickle.dumps(kv)).url == 'memory://'


def test_copy():
  with tempfile.TemporaryDirectory() as dir_path:
    spec = {
        'driver': 'file',
        'path': dir_path,
    }
    t1 = ts.KvStore.open(spec).result()

    t2 = copy.copy(t1)

    assert t1 is not t2

    t3 = copy.deepcopy(t1)

    t1['abc'] = b'def'
    assert t1['abc'] == b'def'
    assert t2['abc'] == b'def'
    assert t3['abc'] == b'def'


def test_keyrange():
  r = ts.KvStore.KeyRange('a', 'b')
  assert repr(r) == "KvStore.KeyRange(b'a', b'b')"


def test_copy_memory():
  spec = {
      'driver': 'memory',
  }
  t1 = ts.KvStore.open(spec).result()

  t2 = copy.copy(t1)

  assert t1 is not t2

  t3 = copy.deepcopy(t1)

  t1['abc'] = b'def'
  assert t1['abc'] == b'def'
  assert t2['abc'] == b'def'
  with pytest.raises(KeyError):
    t3['abc']  # pylint: disable=pointless-statement


def test_copy_range_to_ocdbt_memory():
  context = ts.Context()
  for k in ['a', 'b', 'c']:
    child = ts.KvStore.open(
        {'driver': 'ocdbt', 'base': f'memory://host_{k}/'}, context=context
    ).result()
    child[k] = f'value_{k}'
  parent = ts.KvStore.open(
      {'driver': 'ocdbt', 'base': 'memory://'}, context=context
  ).result()
  for k in ['a', 'b', 'c']:
    child = ts.KvStore.open(
        {'driver': 'ocdbt', 'base': f'memory://host_{k}/'}, context=context
    ).result()
    child.experimental_copy_range_to(parent).result()
  assert parent.list().result() == [b'a', b'b', b'c']


def test_copy_range_to_ocdbt_file():
  context = ts.Context()
  with tempfile.TemporaryDirectory() as dir_path:
    child_spec = {
        'driver': 'ocdbt',
        'base': f'file://{dir_path}/child',
    }
    child = ts.KvStore.open(child_spec, context=context).result()
    for k in ['a', 'b', 'c']:
      child[k] = f'value_{k}'

    parent_spec = {
        'driver': 'ocdbt',
        'base': f'file://{dir_path}',
    }
    parent = ts.KvStore.open(parent_spec, context=context).result()
    child.experimental_copy_range_to(parent).result()

    assert parent.list().result() == [b'a', b'b', b'c']


def test_copy_range_to_memory_fails():
  context = ts.Context()
  child = ts.KvStore.open('memory://child/', context=context).result()
  for k in ['a', 'b', 'c']:
    child[k] = f'value_{k}'
  parent = ts.KvStore.open('memory://', context=context).result()
  with pytest.raises(NotImplementedError):
    child.experimental_copy_range_to(parent).result()


def test_copy_range_to_file_fails():
  context = ts.Context()
  with tempfile.TemporaryDirectory() as dir_path:
    child = ts.KvStore.open(
        f'file://{dir_path}/child/', context=context
    ).result()
    for k in ['a', 'b', 'c']:
      child[k] = f'value_{k}'
    parent = ts.KvStore.open(f'file://{dir_path}', context=context).result()
    with pytest.raises(NotImplementedError):
      child.experimental_copy_range_to(parent).result()


def test_copy_range_to_ocdbt_memory_bad_path():
  context = ts.Context()
  child = ts.KvStore.open(
      {'driver': 'ocdbt', 'base': 'memory://child/'}, context=context
  ).result()
  for k in ['a', 'b', 'c']:
    child[k] = f'value_{k}'
  parent = ts.KvStore.open(
      {'driver': 'ocdbt', 'base': 'memory://c'}, context=context
  ).result()
  with pytest.raises(NotImplementedError):
    child.experimental_copy_range_to(parent).result()
