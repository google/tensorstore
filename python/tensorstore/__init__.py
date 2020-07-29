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
"""TensorStore is a library for reading and writing multi-dimensional arrays."""

import abc as _abc

from ._tensorstore import *
from ._tensorstore import _ContextResource

newaxis = None
"""Alias for `None` used in `indexing expressions<python-indexing>` to specify a

new singleton dimension.
"""


class Indexable(metaclass=_abc.ABCMeta):
  """Abstract base class for types that

  support :ref:`TensorStore indexing operations<python-indexing>`.

  Supported types are:

  - :py:class:`tensorstore.TensorStore`
  - :py:class:`tensorstore.Spec`
  - :py:class:`tensorstore.IndexTransform`
  """


Indexable.register(TensorStore)
Indexable.register(Spec)
Indexable.register(IndexTransform)
