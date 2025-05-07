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
"""Starlark globals for CMake."""

# pylint: disable=invalid-name,missing-function-docstring,relative-beyond-top-level,g-importing-member

from typing import Dict, Optional, Tuple, Type

from .bazel_target import parse_absolute_target
from .bazel_target import TargetId
from .scope_common import ScopeCommon

_BZL_LIBRARIES: Dict[Tuple[TargetId, bool], Type[ScopeCommon]] = {}


def get_bazel_library(
    key: Tuple[TargetId, bool],
) -> Optional[Type[ScopeCommon]]:
  """Returns the target library, if registered."""
  return _BZL_LIBRARIES.get(key)


def register_bzl_library(
    target: str, workspace: bool = False, build: bool = False
):
  target_id = parse_absolute_target(target)

  def register(library: Type[ScopeCommon]):
    if workspace:
      _BZL_LIBRARIES[(target_id, True)] = library
    if build:
      _BZL_LIBRARIES[(target_id, False)] = library
    return library

  return register
