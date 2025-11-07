# Copyright 2025 The TensorStore Authors
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

from .provider import Provider
from .struct import Struct


class FeatureFlagInfo(Provider):
  pass


class LateBoundDefault(Provider):
  __slots__ = ("fragment", "name")

  def __init__(self, fragment, name):
    self.fragment = fragment
    self.name = name

  def __call__(self, ctx: "RuleCtx"):
    # This isn't implemented.
    assert False, "Unimplemented LateBoundDefault"


class BazelModuleConfigCommon:

  FeatureFlagInfo = staticmethod(FeatureFlagInfo)  # type: ignore[not-callable]

  def toolchain_type(self, name, *, mandatory=True, visibility=None):
    del visibility
    return Struct(toolchain_type=name, mandatory=mandatory)

  LateBoundDefault = staticmethod(LateBoundDefault)  # type: ignore[not-callable]

  def configuration_field(self, *, fragment, name):
    return LateBoundDefault(fragment=fragment, name=name)
