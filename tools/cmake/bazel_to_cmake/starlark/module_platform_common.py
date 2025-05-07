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
"""Bazel module: platform_common."""

from .bazel_target import TargetId
from .provider import Provider
from .struct import Struct

# See https://github.com/bazelbuild/bazel/tree/b084956e23a6e50e35fe8886d0f3f7e05f718155/src/main/java/com/google/devtools/build/lib/starlarkbuildapi/platform


class ToolchainInfo(Provider, Struct):

  pass


class ConstraintSettingInfo(Provider):

  __slots__ = ("label", "default_constraint_value")

  def __init__(self, label: TargetId, default_constraint_value: TargetId):
    self.label = label
    self.default_constraint_value = default_constraint_value

  def __repr__(self):
    return (
        f"{self.__class__.__name__}({repr(self.label), repr(self.default_constraint_value)})"
    )


class ConstraintValueInfo(Provider):

  __slots__ = ("constraint", "label")

  def __init__(self, constraint: TargetId, label: TargetId):
    self.constraint = constraint
    self.label = label

  def __repr__(self):
    return (
        f"{self.__class__.__name__}({repr(self.constraint), repr(self.label)})"
    )


class PlatformInfo(Provider):

  __slots__ = ("label", "constraints")

  def __init__(self, label: TargetId, constraints: list[TargetId]):
    self.label = label
    self.constraints = constraints

  def __repr__(self):
    return (
        f"{self.__class__.__name__}({repr(self.label), repr(self.constraints)})"
    )


class TemplateVariableInfo(Provider, Struct):

  def __init__(self, **kwargs):
    super().__init__(variables=kwargs)


class BazelModulePlatformCommon:
  """Bazel module: platform_common."""

  ConstraintSettingInfo = ConstraintSettingInfo
  ConstraintValueInfo = ConstraintValueInfo
  PlatformInfo = PlatformInfo
  TemplateVariableInfo = TemplateVariableInfo
  ToolchainInfo = ToolchainInfo
