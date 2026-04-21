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
"""Implements EvaluationContext."""

import copy
import functools
import inspect
import pathlib
from typing import Any, Callable, Type, TypeVar, cast

from .cmake_builder import CMakeBuilder
from .evaluation_state import EvaluationState
from .package import Package
from .package import Visibility
from .starlark.bazel_target import PackageId
from .starlark.bazel_target import RepositoryId
from .starlark.bazel_target import TargetId
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.provider import TargetInfo
from .starlark.select import Configurable

T = TypeVar("T")
RuleImpl = Callable[[], None]


def trace_exception(f):
  """Decorator adding repr(self) to exceptions."""

  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    try:
      return f(*args, **kwargs)
    except Exception as e:
      e.args = (e.args if e.args else tuple()) + (
          f"from caller {repr(args[0]._caller_package_id)}",
      )
      raise

  return wrapper


class EvaluationContext(InvocationContext):
  """An InvocationContext implementation for EvaluationImpl."""

  __slots__ = (
      "_state",
      "_caller_package_id",
      "_caller_package",
      "_rule_location",
  )

  def __init__(
      self,
      state: EvaluationState,
      package_id: PackageId,
      package: Package | None = None,
  ):
    assert state
    self._state = state
    self._caller_package_id = package_id
    self._caller_package = package
    self._rule_location = ("<unknown>", list())
    assert self._caller_package_id

  def __repr__(self):
    return (
        f"<{self.__class__.__name__}>: "
        "{\n"
        f"  _caller_package_id: {repr(self._caller_package_id)},\n"
        f"  _caller_package: {repr(self._caller_package)},\n"
        f"  _state: {repr(self._state)},\n"
        "}\n"
    )

  def update_current_package(
      self,
      package: Package | None = None,
      package_id: PackageId | None = None,
  ) -> None:
    if package_id is None:
      assert package is not None
      package_id = package.package_id
    self._caller_package_id = package_id
    self._caller_package = package
    assert self._caller_package_id

  # Derived fields
  def snapshot(self) -> "EvaluationContext":
    return copy.copy(self)

  def access(self, provider_type: Type[T]) -> T:
    if provider_type == EvaluationState:
      return cast(T, self._state)
    elif provider_type == CMakeBuilder:
      return cast(T, self._state.builder)
    elif provider_type == Package:
      assert self._caller_package
      return cast(T, self._caller_package)
    elif provider_type == Visibility:
      assert self._caller_package
      return cast(T, Visibility(self._caller_package))
    # We can't import EvaluationImpl here due to circular dependency.
    return super().access(provider_type)

  @property
  def caller_package(self) -> Package | None:
    return self._caller_package

  @property
  def caller_package_id(self) -> PackageId:
    return self._caller_package_id

  def record_rule_location(self, mnemonic):
    # Record the path of non-python callers.
    s = inspect.stack()
    callers = []
    for i in range(2, min(6, len(s))):
      c = inspect.getframeinfo(s[i][0])
      if not c.filename.endswith(".py"):
        callers.append(f"{c.filename}:{c.lineno}")
    self._rule_location = (mnemonic, callers)

  def resolve_source_root(self, repository_id: RepositoryId) -> pathlib.Path:
    return pathlib.Path(
        self._state.workspace.all_repositories[repository_id].source_directory
    )

  def resolve_output_root(
      self, repository_id: RepositoryId
  ) -> pathlib.PurePosixPath:
    return self._state.workspace.all_repositories[
        repository_id
    ].cmake_binary_dir

  @trace_exception
  def apply_repo_mapping(
      self, target_id: TargetId, mapping_repository_id: RepositoryId | None
  ) -> TargetId:
    # Resolve repository mappings
    if mapping_repository_id is None:
      assert self._caller_package_id
      mapping_repository_id = self._caller_package_id.repository_id

    mapping_repo = self._state.workspace.all_repositories.get(
        mapping_repository_id
    )
    assert mapping_repo is not None
    target = mapping_repo.apply_repo_mapping(target_id)

    # Resolve --bind in the active repo.
    while target in self._state.active_repo.bindings:
      target = self._state.active_repo.bindings[target]

    if self._state.workspace.verbose and target != target_id:
      print(
          f"apply_repo_mapping({target_id.as_label()}) => {target.as_label()}"
      )

    return target

  def load_library(self, target: TargetId) -> dict[str, Any]:
    return self._state.load_library(target)

  @trace_exception
  def get_target_info(self, target_id: TargetId) -> TargetInfo:
    return self._state.get_target_info(target_id)

  def add_rule(
      self,
      rule_id: TargetId,
      impl: RuleImpl,
      outs: list[TargetId] | None = None,
      visibility: list[RelativeLabel] | None = None,
      **kwargs,
  ) -> None:
    if visibility is not None:
      if kwargs.get("analyze_by_default") is None:
        assert self._caller_package
        kwargs["analyze_by_default"] = Visibility(
            self._caller_package
        ).analyze_by_default(self.resolve_target_or_label_list(visibility))

    self._state.add_rule_impl(
        self._rule_location[0],
        self._rule_location[1],
        rule_id=rule_id,
        impl=impl,
        outs=outs,
        **kwargs,
    )

  @trace_exception
  def add_analyzed_target(self, target_id: TargetId, info: TargetInfo) -> None:
    self._state.add_analyzed_target(target_id, info)

  def evaluate_condition(self, target_id: TargetId) -> bool:
    return self._state.evaluate_condition(target_id)

  def evaluate_configurable(self, configurable: Configurable[T]) -> T:
    return self._state.evaluate_configurable(configurable)

  def set_module_info(
      self, name: str, version: str, compatibility_level: int, repo_name: str
  ) -> None:
    self._state.set_module_info(name, version, compatibility_level, repo_name)

  def add_bazel_dep(
      self,
      name: str,
      version: str,
      max_compatibility_level: int,
      repo_name: str,
      dev_dependency: bool,
  ) -> None:
    self._state.add_bazel_dep(
        name,
        version,
        max_compatibility_level,
        repo_name,
        dev_dependency,
    )

  def use_repo(self, extension_proxy, *args, **kwargs) -> None:
    self._state.use_repo(extension_proxy, *args, **kwargs)

  def add_module_override(
      self, module_name: str, override_info: dict[str, Any]
  ) -> None:
    self._state.add_module_override(module_name, override_info)

  def include_module_file(
      self, label: RelativeLabel, scope: Any = None
  ) -> None:
    self._state.include_module_file(label, scope)
