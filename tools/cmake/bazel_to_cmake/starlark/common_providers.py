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
"""Defines information about analyzed Bazel targets."""

# pylint: disable=missing-function-docstring,relative-beyond-top-level

import pathlib
from typing import Any, Iterable, Optional, TypeVar

from .bazel_target import TargetId
from .provider import Provider


PathLike = TypeVar(
    "PathLike", str, pathlib.Path, pathlib.PurePath, pathlib.PurePosixPath
)


class BuildSettingProvider(Provider):
  """Build setting value (i.e. flag value) corresponding to a Bazel target."""

  __slots__ = ("value",)

  def __init__(self, value: Any):
    self.value = value

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.value)})"


class ConditionProvider(Provider):
  """Condition value corresponding to a Bazel target."""

  __slots__ = ("value",)

  def __init__(self, value: bool):
    self.value = value

  def __repr__(self):
    return f"{self.__class__.__name__}({self.value})"


class FilesProvider(Provider):
  """Files corresponding to a Bazel target."""

  __slots__ = ("paths",)

  def __init__(self, paths: Iterable[PathLike]):
    self.paths = [str(x) for x in paths]

  def __repr__(self):
    return f"{self.__class__.__name__}({repr(self.paths)})"


class ProtoLibraryProvider(Provider):
  __slots__ = (
      "bazel_target",
      "srcs",
      "deps",
      "strip_import_prefix",
      "import_prefix",
  )

  def __init__(
      self,
      bazel_target: TargetId,
      srcs: Iterable[TargetId],
      deps: Iterable[TargetId],
      strip_import_prefix: Optional[str],
      import_prefix: Optional[str],
  ):
    self.bazel_target = bazel_target
    self.srcs = set(srcs)
    self.deps = set(deps)
    self.strip_import_prefix = strip_import_prefix
    self.import_prefix = import_prefix

  def __repr__(self):
    return (
        f"{self.__class__.__name__}("
        f"{repr(self.bazel_target)},"
        f"{repr(self.srcs)},"
        f"{repr(self.deps)},"
        f"{repr(self.strip_import_prefix)},"
        f"{repr(self.import_prefix)})"
    )


# See https://github.com/bazelbuild/bazel/tree/b084956e23a6e50e35fe8886d0f3f7e05f718155/src/main/java/com/google/devtools/build/lib/starlarkbuildapi/platform
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
