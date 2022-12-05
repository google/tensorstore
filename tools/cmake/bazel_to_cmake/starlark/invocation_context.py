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

# pylint: disable=missing-function-docstring,relative-beyond-top-level,g-importing-member

import pathlib
from .select import Configurable
from .select import Select
from .select import SelectExpression
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, cast

from .bazel_target import PackageId
from .bazel_target import RepositoryId
from .bazel_target import TargetId
from .common_providers import BuildSettingProvider
from .common_providers import ConditionProvider
from .label import Label
from .label import RelativeLabel
from .provider import TargetInfo

RuleImpl = Callable[[], None]
T = TypeVar("T")


class InvocationContext(object):
  """InvocationContext provides access to the the currently evaluating package.
  """

  def __repr__(self):
    return f"<{self.__class__.__name__}>: {self.__dict__}"

  def __str__(self):
    raise AttributeError("InvocationContext -> string conversion not allowed")

  def access(self, provider_type: Type[T]) -> T:
    raise ValueError(f"Type {provider_type.__name__} unavailable.")

  def snapshot(self) -> "InvocationContext":
    raise NotImplementedError("snapshot")

  @property
  def caller_repository_name(self) -> str:
    return f"@{self.caller_package_id.repository_name}//{self.caller_package_id.package_name}"

  @property
  def caller_package_id(self) -> PackageId:
    raise NotImplementedError("caller_package_id")

  def resolve_source_root(self, repository_id: RepositoryId) -> str:
    # This should return something like the Package.source_directory
    raise NotImplementedError("resolve_source_root")

  def resolve_output_root(self, repository_id: RepositoryId) -> str:
    # This should return something like the Package.cmake_root
    raise NotImplementedError("resolve_output_root")

  def resolve_repo_mapping(
      self, target_id: TargetId,
      mapping_repository_id: Optional[RepositoryId]) -> TargetId:
    raise NotImplementedError("resolve_repo_mapping")

  def load_library(self, target: TargetId) -> Dict[str, Any]:
    raise NotImplementedError("load_library")

  def get_target_info(self, target_id: TargetId) -> TargetInfo:
    raise NotImplementedError("get_target_info")

  def add_rule(self,
               rule_id: TargetId,
               impl: RuleImpl,
               outs: Optional[List[TargetId]] = None,
               **kwargs) -> None:
    """Adds a rule.

    Args:
      rule_id: TargetId for the rule.
      impl: Implementation function called during analysis phase.
      outs: Additional output targets beyond `rule_label` itself.
      analyze_by_default: Whether to analyze by default, as opposed to only if
        it is a dependency of another target being analyzed.
    """
    raise NotImplementedError("add_rule")

  def add_analyzed_target(self, target_id: TargetId, info: TargetInfo) -> None:
    """Adds the `TargetInfo' for an analyzed target.

    This must be called by the `RuleImpl` function for the rule_label and each
    output target.

    Args:
      target_id: The rule target identifier.
      info: The analyzed TargetInfo associated with the target_id.
    """
    raise NotImplementedError("add_analyzed_target")

  def evaluate_condition(self, target_id: TargetId) -> bool:
    return self.get_target_info(target_id)[ConditionProvider].value

  def evaluate_build_setting(self, target_id: TargetId) -> Any:
    return self.get_target_info(target_id)[BuildSettingProvider].value

  def evaluate_configurable(self, configurable: Configurable[T]) -> T:
    """Evaluates a `Configurable` expression."""
    assert configurable is not None
    if isinstance(configurable, Select) or isinstance(configurable,
                                                      SelectExpression):
      return cast(T, configurable.evaluate(self.evaluate_condition))
    return cast(T, configurable)

  def evaluate_configurable_list(
      self,
      configurable: Optional[Configurable[List[T]]],
  ) -> List[T]:
    if configurable is None:
      return []
    if isinstance(configurable, list):
      # This occurs when a single configurable is put into a list, as happens
      # in the bazel_skylib.copy_file rule.
      return cast(List[T],
                  [self.evaluate_configurable(x) for x in configurable])
    else:
      evaluated: List[T] = self.evaluate_configurable(configurable)
      assert isinstance(evaluated, list)
      return evaluated

  def resolve_target(
      self,
      label_string: str,
      mapping_repository_id: Optional[RepositoryId] = None) -> TargetId:
    # Use package-level resolution.
    assert label_string
    assert not isinstance(label_string, list)
    assert not isinstance(label_string, TargetId)
    return self.resolve_repo_mapping(
        self.caller_package_id.parse_target(label_string),
        mapping_repository_id)

  def resolve_target_or_label(
      self,
      target: RelativeLabel,
      mapping_repository_id: Optional[RepositoryId] = None) -> TargetId:
    assert target
    assert not isinstance(target, TargetId)
    if isinstance(target, Label):
      return target.target_id

    return self.resolve_target(str(target), mapping_repository_id)

  def resolve_target_or_label_list(
      self,
      targets: List[RelativeLabel],
      mapping_repository_id: Optional[RepositoryId] = None) -> List[TargetId]:
    if targets is None:
      return []
    assert isinstance(targets, list)
    return [
        self.resolve_target_or_label(t, mapping_repository_id) for t in targets
    ]

  def get_source_package_dir(self, package_id: PackageId) -> str:
    assert isinstance(package_id, PackageId)
    return str(
        pathlib.PurePosixPath(
            self.resolve_source_root(package_id.repository_id)).joinpath(
                package_id.package_name))

  def get_source_file_path(self, target_id: TargetId) -> Optional[str]:
    assert isinstance(target_id, TargetId)
    return str(
        pathlib.PurePosixPath(
            self.resolve_source_root(target_id.repository_id)).joinpath(
                target_id.package_name, target_id.target_name))

  def get_generated_file_path(self, target_id: TargetId) -> str:
    assert isinstance(target_id, TargetId)
    root = self.resolve_output_root(target_id.repository_id)
    if root is None:
      raise ValueError(
          f"Target '{target_id.as_label()}' missing output root directory.")
    return str(
        pathlib.PurePosixPath(root).joinpath(target_id.package_name,
                                             target_id.target_name))
