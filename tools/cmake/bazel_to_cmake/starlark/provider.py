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
"""starlark Provider type.

https://bazel.build/rules/lib/globals#provider
"""

# pylint: disable=missing-function-docstring,relative-beyond-top-level,missing-class-docstring

from typing import TypeVar, Type, Optional, cast


class Provider:
  __slots__ = ()


P = TypeVar('P', bound=Provider)


class TargetInfo:
  """Providers associated with an analyzed Bazel target."""
  __slots__ = ('_providers',)

  def __init__(self, *args: Provider):
    providers = {}
    for p in args:
      providers[type(p)] = p
    self._providers = providers

  def __getitem__(self, provider_type: Type[P]) -> P:
    return cast(P, self._providers[provider_type])

  def get(self, provider_type: Type[P]) -> Optional[P]:
    return cast(Optional[P], self._providers.get(provider_type))

  def __iter__(self):
    return iter(self._providers.values())

  def __repr__(self):
    return ('{' + ', '.join(repr(value) for value in self._providers.values()) +
            '}')


class GenericProvider(Provider):

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

  def __eq__(self, other):
    if not isinstance(other, Provider):
      return False
    if self._fields != other._fields:
      return False
    for x in self._fields:
      if self.__dict__.get(x) != other.__dict__.get(x):
        return False
    return True

  def __ne__(self, other):
    if not isinstance(other, Provider):
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


def _make_provider(fields, **kwargs):
  if fields is not None:
    # Validate kwargs
    for k in kwargs:
      if k not in fields:
        raise AttributeError(f'field {k} not allowed in provider')
  return GenericProvider(**kwargs)


def provider(doc='', *, fields=None, init=None):
  """Returns a Starlark 'provider' callable."""
  del doc
  provider_lambda = lambda **kwargs: _make_provider(fields, **kwargs)

  # Maybe instantiate the type.
  if init is None:
    return provider_lambda

  def ctor(**kwargs):
    return provider_lambda(**init(**kwargs))

  return ctor, provider_lambda
