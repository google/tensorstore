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
"""CMake Provider types."""

from typing import NamedTuple, Optional


class CMakePackage(str):
  pass


class CMakeTarget(str):
  pass


class CMakeTargetPair(NamedTuple):
  """CMakeTargetPair identifies a cmake target, optionally with an alias."""

  cmake_package: Optional[CMakePackage]
  target: CMakeTarget
  alias: Optional[CMakeTarget] = None

  def with_alias(self, alias: Optional[CMakeTarget]) -> "CMakeTargetPair":
    if alias is not None:
      assert isinstance(alias, CMakeTarget)
    return self._replace(alias=alias)

  @property
  def dep(self) -> CMakeTarget:
    return self.alias or self.target

  def __str__(self) -> str:
    raise NotImplementedError
