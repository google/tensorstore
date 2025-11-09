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

from typing import Optional, Tuple, Type, TypeVar, cast

from .struct import Struct


class Provider:
  __slots__ = ()


P = TypeVar('P', bound=Provider)
ProviderTuple = Tuple[Provider, ...]


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

  def __tuple__(self):
    return tuple(self._providers.values())

  def __repr__(self):
    return (
        '{' + ', '.join(repr(value) for value in self._providers.values()) + '}'
    )


def _make_provider(cls, fields, **kwargs):

  if fields is not None:
    # Validate kwargs
    for k in kwargs:
      if k not in fields:
        raise AttributeError(f'field {k} not allowed in provider')

  return cls(**kwargs)


def provider(doc='', *, fields=None, init=None):
  """Returns a Starlark 'provider' callable."""
  del doc

  class GenericProvider(Provider, Struct):
    pass

  provider_lambda = lambda **kwargs: _make_provider(
      GenericProvider, fields, **kwargs
  )

  # Maybe instantiate the type.
  if init is None:
    return provider_lambda

  def ctor(**kwargs):
    return provider_lambda(**init(**kwargs))

  return ctor, provider_lambda
