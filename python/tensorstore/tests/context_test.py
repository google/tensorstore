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
"""Tests for tensorstore.Context."""

import pickle

import pytest
import tensorstore as ts


def test_context_spec():
  json_spec = {
      'memory_key_value_store': {},
      'memory_key_value_store#a': 'memory_key_value_store',
      'memory_key_value_store#b': {},
  }

  spec = ts.ContextSpec(json_spec)

  assert spec.to_json() == json_spec

  assert repr(spec) == '''ContextSpec({
  'memory_key_value_store': {},
  'memory_key_value_store#a': 'memory_key_value_store',
  'memory_key_value_store#b': {},
})'''

  new_spec = pickle.loads(pickle.dumps(spec))
  assert new_spec.to_json() == json_spec


def test_pickle():

  parent_context = ts.Context({
      'memory_key_value_store#c': {},
  })

  json_spec = {
      'memory_key_value_store': {},
      'memory_key_value_store#a': 'memory_key_value_store',
      'memory_key_value_store#b': {},
  }
  child_spec = ts.ContextSpec(json_spec)

  context = ts.Context(child_spec, parent_context)

  with pytest.raises(ValueError,
                     match="Invalid context resource identifier: \"abc\""):
    context['abc']

  assert context.spec is child_spec
  assert context.parent is parent_context
  assert parent_context.parent is None
  assert context['memory_key_value_store'].to_json() == {}
  assert repr(context['memory_key_value_store']) == '_ContextResource({})'
  assert context['memory_key_value_store#b'].to_json() == {}
  assert context['memory_key_value_store#c'].to_json() == {}
  assert context['memory_key_value_store'] is context['memory_key_value_store#a']
  assert context['memory_key_value_store'] is not context[
      'memory_key_value_store#b']

  # Ensure that we can pickle a normally constructed Context, and also that we
  # can repickle a previously unpickled Context.
  for i in range(3):
    res_a, res_b, res_c, new_ctx, new_parent = pickle.loads(
        pickle.dumps([
            context['memory_key_value_store#a'],
            context['memory_key_value_store#b'],
            context['memory_key_value_store#c'],
            context,
            parent_context,
        ]))

    assert res_a is not res_b
    assert res_a.to_json() == {}
    assert res_b.to_json() == {}
    assert new_ctx.spec.to_json() == json_spec
    assert new_parent.spec.to_json() == parent_context.spec.to_json()
    assert new_ctx['memory_key_value_store'] is res_a
    assert new_ctx['memory_key_value_store#a'] is res_a
    assert new_ctx['memory_key_value_store#b'] is res_b
    assert new_ctx['memory_key_value_store#c'] is res_c
    assert new_parent['memory_key_value_store#c'] is res_c
    assert new_parent['memory_key_value_store'] is not new_ctx[
        'memory_key_value_store']
    # Verify that the relationship between `parent` and `context` is preserved in
    # `new_parent` and `new_ctx`.
    assert new_ctx.parent is new_parent

    # Note that the unpickled objects do not have the same identity as the
    # original objects (in general unpickling may occur in a separate process).
    assert new_parent is not parent_context
    assert new_ctx is not context
    parent_context, context = new_parent, new_ctx
