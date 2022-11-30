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
"""starlark depset function.

https://bazel.build/rules/lib/depset
https://bazel.build/rules/lib/globals#depset
"""

# pylint: disable=missing-function-docstring

import itertools


class DepSet(set):

  def __init__(self, direct=None, order='default', transitive=None):
    """Creates a depset.

    Args:
      direct: The direct elements to add to the depset.
      order: The order of elements in this depset. Not supported by the
          testing framework.
      transitive: A list of depsets to unroll into the new depset.

    Returns:
      The new depset.
    """
    # Ignore the order parameter. Order is not guaranteed to be the same as
    # Starlark's depsets' ordering.
    del order
    if direct is None:
      direct = []
    if transitive is None:
      transitive = []

    super().__init__(itertools.chain(direct, *transitive))

  def __add__(self, other):
    return self.__class__(direct=self.union(other))

  # pylint: disable=invalid-name
  def to_list(self):
    return list(self)

  def __repr__(self):
    return f'depset({repr(list(self))})'


def depset(direct=None, **kwargs):
  return DepSet(direct=direct, **kwargs)
