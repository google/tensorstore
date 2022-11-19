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

import struct as skylark_struct


def _make_provider(fields, **kwargs):
  if fields is not None:
    # Validate kwargs
    for k in kwargs:
      if k not in fields:
        raise AttributeError(f"field {k} not allowed in provider")
  return skylark_struct.struct(**kwargs)


def provider(doc="", *args, fields=None, init=None):
  """Returns a Starlark 'provider' callable."""

  # Only the doc field may be positional
  if args:
    raise ValueError("bad provider() call")

  provider_lambda = lambda **kwargs: _make_provider(fields, **kwargs)

  # Maybe instantiate the type.
  if init is None:
    return provider_lambda

  def ctor(**kwargs):
    return provider_lambda(**init(**kwargs))

  return ctor, provider_lambda
