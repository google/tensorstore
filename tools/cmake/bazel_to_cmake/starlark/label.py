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
"""Implement the Label type for bazel."""

from typing import Union, Callable

from .bazel_target import RepositoryId
from .bazel_target import TargetId

RelativeLabel = Union[str, "Label"]


# There are a few complications with resolving relative labels in both the
# constructor and Label.relative function.
#
# Consider the following file:
#  x.bzl
#    def resolve(x: Label, y:str): return x.relative(y)
#    def get(x: str): return Label(x)
#
# And the WORKSPACE defines a repo mapping:
#   workspace_name = "mapped", repo_mapping = { "@foo" : "@bar" }
#   workspace_name = "unmapped", repo_mapping = { }
#
# The resolve call
# Then in a build file:
#   load("@unmapped//:x.bzl", unmapped_resolve="resolve", unmapped_get="get")
#   load("@mapped//:x.bzl", mapped_resolve="resolve", mapped_get="get")
#
#   print(unmapped_get("@foo//:x")) => "@foo//:x
#   print(mapped_get("@foo//:x")) => "@bar//:x
#
#   print(unmapped_relative(Label("//:x"), "@foo//:x")) => "@foo//:x
#   print(mapped_relative(Label("//:x"), "@foo//:x")) => "@bar//:x
#
class Label:
  """Corresponds to the Bazel `Label` type.

    This holds a reference to the `InvocationContext` in order to compute
    `workspace_root`.
  """
  __slots__ = ("target_id", "_resolve_workspace")

  def __init__(self, target_id: TargetId,
               resolve_workspace: Callable[[RepositoryId], str]):
    # Rely on the constructor function to apply the the repository mapping.
    self.target_id = target_id
    self._resolve_workspace = resolve_workspace

  @property
  def workspace_name(self) -> str:
    return self.target_id.repository_name

  @property
  def package(self) -> str:
    return self.target_id.package_name

  @property
  def name(self) -> str:
    return self.target_id.target_name

  def __str__(self) -> str:
    return f"{self.target_id.as_label()}"

  def __repr__(self) -> str:
    return f"Label(\"{self}\")"

  @property
  def workspace_root(self) -> str:
    # In bazel, the workspace root is always something like
    # external/workspace_name, however for the main repository the
    # workspace_name is empty, and so is workspace_root.
    return self._resolve_workspace(self.target_id.repository_id)

  def relative(self, label_str: str) -> "Label":
    raise ValueError("Label.relative() is not correct")
    # TODO: Remapping should happen via the macro definition scope
    # (e.g. caller.globals._target_id), not the constructor's scope.
    return Label(self.target_id.parse_label(label_str), self._resolve_workspace)


def as_target_id(label: Union[Label, TargetId]) -> TargetId:
  assert not isinstance(label, str)
  if isinstance(label, Label):
    return label.target_id
  elif isinstance(label, TargetId):
    return label
  else:
    raise ValueError(f"{type(label)} must be Label or TargetId")
