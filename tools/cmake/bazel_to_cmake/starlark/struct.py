# Copyright 2022 The TensorStore Authors
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
"""starlark struct function.

https://bazel.build/rules/lib/struct
"""

# pylint: disable=missing-function-docstring


class Struct(object):

  def __init__(self, **kwargs):
    self.__dict__['_hidden_fields'] = sorted(kwargs.keys())
    self.__dict__.update(kwargs)

  @property
  def _fields(self):
    return self.__dict__['_hidden_fields']

  def __repr__(self):
    kwargs_repr = ','.join(
        f'{k}={repr(self.__dict__.get(k))}' for k in self._fields)
    return f'struct({kwargs_repr})'

  def __add__(self, addend: 'Struct'):
    """In Starlark, struct + struct = struct."""
    common_fields = set(self._fields) & set(addend._fields)
    if common_fields:
      raise ValueError('Cannot concat structs with common field(s): {}'.format(
          ', '.join(common_fields)))

    fields = {k: self.__dict__.get(k) for k in self._fields}
    fields.update({k: addend.__dict__.get(k) for k in addend._fields})
    return self.__class__(**fields)

  def __eq__(self, other):
    if not isinstance(other, Struct):
      return False
    if self._fields != other._fields:
      return False
    for x in self._fields:
      if self.__dict__.get(x) != other.__dict__.get(x):
        return False
    return True

  def __ne__(self, other):
    if not isinstance(other, Struct):
      return True
    if self._fields != other._fields:
      return True
    for x in self._fields:
      if self.__dict__.get(x) != other.__dict__.get(x):
        return True
    return False

  def __setattr__(self, *ignored):
    raise NotImplementedError

  def __delattr__(self, *ignored):
    raise NotImplementedError
