# Copyright 2023 The TensorStore Authors
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

import pickle

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
