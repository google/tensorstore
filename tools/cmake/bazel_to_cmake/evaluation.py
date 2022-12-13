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
"""Starlark evaluation for CMake.

Similar to the real Bazel, evaluation is performed in several phases:

0. A "CMakeLists.txt" invokes `bazel_to_cmake` via the CMake `execute_process`
   command.  All necessary configuration variables are provided to
   `bazel_to_cmake`.

1. [Top-level] Workspace loading: the top-level WORKSPACE and any bzl libraries
   loaded by it are evaluated.  Note: Unlike Bazel, there is currently no real
   support for repository rules, and therefore workspace loading is just a
   single phase.  Third-party dependencies are emitted as calls to
   `FetchContent` CMake commands.  For dependencies for which `bazel_to_cmake`
   is enabled, a `CMakeLists.txt` is generated that invokes `bazel_to_cmake`
   again.

2. [Top-level] Package loading phase: the BUILD files and any bzl libraries
   loaded by them are evaluated.  Rules call `EvaluationContext.add_rule` to
   register rules to be evaluated during the analysis phase.

3. [Top-level] Analysis phase: The rules added during the loading phase are
   evaluated, which:

   - results in providers for targets being set by calls to
     `EvaluationContext.add_analyzed_target`, and

   - generates CMake code as a side effect.

   If not the top-level CMake project (not necessary the same as the top-level
   `bazel_to_cmake` repo), only transitive dependencies of public, non-test
   targets are analyzed.

4. [Top-level] Output phase: The CMake code is written to `build_rules.cmake`,
   the `Workspace` is saved in pickle format, and `bazel_to_cmake` exits.

5. The CMake code is evaluated by CMake.

6. For dependencies for which `bazel_to_cmake` is enabled, `bazel_to_cmake` is
   invoked again.

7. [Sub-project] The `Workspace` saved from the top-level invocation of
   `bazel_to_cmake` is loaded.  The workspace loading phase (step 1) is skipped.

7. [Sub-project] Package loading phase: the BUILD files and any bzl libraries
   loaded by them are evaluated.  Rules add targets that remain unanalyzed.

8. [Sub-project] Analysis phase: The targets added during the loading phase are
   evaluated, which generates CMake code as a side effect.

9. [Sub-project] Output phase: The CMake code is written to `build_rules.cmake`
   and `bazel_to_cmake` exits.

10. The CMake code is evaluated by CMake.
"""

# pylint: disable=relative-beyond-top-level,protected-access,missing-function-docstring,invalid-name,g-doc-args,g-doc-return-or-yield

import copy
import enum
import inspect
import os
import pathlib
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple, Type, TypeVar, cast

from . import cmake_builder
from .cmake_builder import CMakeBuilder
from .cmake_target import CMakeDepsProvider
from .cmake_target import CMakePackageDepsProvider
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetPair
from .cmake_target import CMakeTargetPairProvider
from .cmake_target import label_to_generated_cmake_target
from .package import Package
from .package import Visibility
from .starlark.bazel_globals import BazelWorkspaceGlobals
from .starlark.bazel_globals import BuildFileGlobals
from .starlark.bazel_globals import BuildFileLibraryGlobals
from .starlark.bazel_globals import get_bazel_library
from .starlark.bazel_target import PackageId
from .starlark.bazel_target import remap_target_repo
from .starlark.bazel_target import RepositoryId
from .starlark.bazel_target import TargetId
from .starlark.common_providers import BuildSettingProvider
from .starlark.common_providers import ConditionProvider
from .starlark.common_providers import FilesProvider
from .starlark.ignored import IgnoredLibrary
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel
from .starlark.provider import TargetInfo
from .starlark.select import Configurable
from .starlark.select import Select
from .starlark.select import SelectExpression
from .util import cmake_is_true
from .workspace import Repository
from .workspace import Workspace

T = TypeVar("T")

RuleImpl = Callable[[], None]


class RuleInfo(NamedTuple):
  outs: List[TargetId]
  impl: RuleImpl
  kind: Optional[str]


class Phase(enum.Enum):
  LOADING_WORKSPACE = 1
  LOADING_BUILD = 2
  ANALYZE = 3


def _get_kind(currentframe) -> Optional[str]:
  if not currentframe:
    return None
  kind = currentframe.f_back.f_back.f_code.co_name
  if kind.startswith("bazel_"):
    kind = kind[len("bazel_"):]
  return kind


class EvaluationState:
  """State used while evaluating Starlark code."""

  def __init__(self, repo: Repository):
    self.repo = repo
    self.workspace: Workspace = repo.workspace
    self.builder = CMakeBuilder()
    self._evaluation_context = EvaluationContext(
        self, self.repo.repository_id.get_package_id(""))
    self.loaded_files: Set[str] = set()
    self._loaded_libraries: Dict[Tuple[TargetId, bool], Dict[str, Any]] = dict()
    self._wrote_dummy_source = False
    self.required_dep_packages: Set[str] = set()
    self.errors: List[str] = []
    # Tracks CMakeTargetPairs provided by get_dep() which don't have
    # corresponding rules in this repo.
    self._cmake_dep_pairs: Dict[TargetId, CMakeTargetPair] = {}
    # Maps targets to their rules.
    self._unanalyzed_rules: Set[TargetId] = set()
    self._all_rules: Dict[TargetId, RuleInfo] = {}
    self._unanalyzed_targets: Dict[TargetId, TargetId] = {}
    self._targets_to_analyze: Set[TargetId] = set()
    self._analyzed_targets: Dict[TargetId, TargetInfo] = {}
    self._call_after_workspace_loading: List[Callable[[], None]] = []
    self._call_after_analysis: List[Callable[[], None]] = []
    self.public_only = not (self.repo.top_level and cmake_is_true(
        self.workspace.cmake_vars["PROJECT_IS_TOP_LEVEL"]))
    self._phase: Phase = Phase.LOADING_WORKSPACE
    self._verbose = self.workspace._verbose

  @property
  def targets_to_analyze(self) -> List[TargetId]:
    return sorted(self._targets_to_analyze)

  def analyze(self, targets: List[TargetId]):
    """Analayze the transitive dependencies of `targets`.

    Note that any targets that are skipped will not be available for use as
    dependencies of targets defined in other repositories.
    """
    self._phase = Phase.ANALYZE
    for target in targets:
      if self._verbose:
        print(f"Analyze: {target.as_label()}")
      assert isinstance(target, TargetId)
      self.get_target_info(target)

    for package in sorted(self.required_dep_packages):
      self.builder.find_package(
          package, section=cmake_builder.FIND_DEP_PACKAGE_SECTION)

    for callback in self._call_after_analysis:
      callback()

  def add_rule(self,
               rule_id: TargetId,
               impl: RuleImpl,
               outs: Optional[List[TargetId]] = None,
               analyze_by_default: bool = True) -> None:
    """Adds a rule.

    Args:
      rule_label: Label for the rule.
      impl: Implementation function called during analysis phase.
      outs: Additional output targets beyond `rule_label` itself.
      analyze_by_default: Whether to analyze by default, as opposed to only if
        it is a dependency of another target being analyzed.
    """
    assert isinstance(rule_id, TargetId), f"Requires TargetId: {repr(rule_id)}"
    if rule_id in self._all_rules:
      raise ValueError(f"Duplicate rule: {rule_id.as_label()}")
    if outs is None:
      outs = []

    # kind is assigned from caller function name
    kind = _get_kind(inspect.currentframe())
    r = RuleInfo(outs, impl, kind)
    self._all_rules[rule_id] = r
    self._unanalyzed_rules.add(rule_id)
    for out_id in r.outs:
      if (out_id in self._unanalyzed_targets or
          out_id in self._analyzed_targets):
        raise ValueError(f"Duplicate output: {out_id.as_label()}")
      self._unanalyzed_targets[out_id] = rule_id
    self._unanalyzed_targets[rule_id] = rule_id
    if analyze_by_default:
      self._targets_to_analyze.add(rule_id)

  def add_analyzed_target(self, target_id: TargetId, info: TargetInfo) -> None:
    """Adds the `TargetInfo' for an analyzed target.

    This must be called by the `RuleImpl` function for the rule_label and each
    output target.
    """
    assert isinstance(target_id,
                      TargetId), f"Requires TargetId: {repr(target_id)}"
    assert info is not None
    self._analyzed_targets[target_id] = info

    if info.get(CMakeTargetPairProvider) is not None:
      self._cmake_dep_pairs.pop(target_id, None)

  def visit_analyzed_targets(self, visitor: Callable[[TargetId, TargetInfo],
                                                     None]):
    for target, info in self._analyzed_targets.items():
      visitor(target, info)

  def visit_cmake_dep_pairs(self, visitor: Callable[[TargetId, CMakeTargetPair],
                                                    None]):
    for target, info in self._cmake_dep_pairs.items():
      visitor(target, info)

  def get_optional_target_info(self,
                               target_id: TargetId) -> Optional[TargetInfo]:
    assert isinstance(target_id,
                      TargetId), f"Requires TargetId: {repr(target_id)}"
    analyzed_targets = self._analyzed_targets
    info = analyzed_targets.get(target_id)
    if info is not None:
      return info

    unanalyzed_targets = self._unanalyzed_targets
    if target_id not in unanalyzed_targets:
      # Is this a global persistent target?
      info = self.workspace._persisted_target_info.get(target_id)
      if info is not None:
        return info
      # Is this a source file?
      source_path = self.get_source_file_path(target_id)
      if source_path is None or not os.path.isfile(source_path):
        return None
      info = TargetInfo(FilesProvider([source_path]))
      analyzed_targets[target_id] = info
      return info

    rule_id = unanalyzed_targets.get(target_id)
    assert rule_id is not None
    rule_info = self._all_rules.get(rule_id, None)
    if rule_info is None:
      raise ValueError(f"Error analyzing {rule_id.as_label()}: Not found")
    try:
      self._unanalyzed_rules.remove(rule_id)
      rule_info.impl()
      for out_id in rule_info.outs:
        unanalyzed_targets.pop(out_id, None)
        assert out_id in analyzed_targets
      unanalyzed_targets.pop(rule_id)
      assert rule_id in analyzed_targets
    except Exception as e:
      raise ValueError(
          f"Error analyzing {rule_id.as_label()}  with {rule_info}") from e
    return analyzed_targets[target_id]

  def get_target_info(self, target_id: TargetId) -> TargetInfo:
    assert isinstance(target_id, TargetId)
    info = self.get_optional_target_info(target_id)
    if info is None:
      raise ValueError(f"Target not found: {target_id.as_label()}")
    return info

  def get_source_file_path(self, target_id: TargetId) -> Optional[str]:
    assert isinstance(target_id, TargetId)
    repo = self.workspace.repos.get(target_id.repository_id)
    if repo is None:
      return None
    return str(
        pathlib.PurePosixPath(repo.source_directory).joinpath(
            target_id.package_name, target_id.target_name))

  def get_generated_file_path(self, target_id: TargetId) -> str:
    assert isinstance(target_id, TargetId)
    repo = self.workspace.repos.get(target_id.repository_id)
    if repo is None:
      raise ValueError(
          f"Unknown repository name in target: {target_id.as_label()}")
    return str(
        pathlib.PurePosixPath(repo.cmake_binary_dir).joinpath(
            target_id.package_name, target_id.target_name))

  def evaluate_condition(self, target_id: TargetId) -> bool:
    assert isinstance(target_id, TargetId)
    assert self._phase == Phase.ANALYZE
    return self.get_target_info(target_id)[ConditionProvider].value

  def evaluate_build_setting(self, target_id: TargetId) -> Any:
    assert isinstance(target_id, TargetId)
    assert self._phase == Phase.ANALYZE
    return self.get_target_info(target_id)[BuildSettingProvider].value

  def evaluate_configurable(self, configurable: Configurable[T]) -> T:
    """Evaluates a `Configurable` expression."""
    assert self._phase == Phase.ANALYZE
    if isinstance(configurable, Select) or isinstance(configurable,
                                                      SelectExpression):
      return configurable.evaluate(self.evaluate_condition)
    return configurable

  def get_file_paths(
      self,
      target: TargetId,
      custom_target_deps: Optional[List[CMakeTarget]],
  ) -> List[str]:
    info = self.get_target_info(target)
    if custom_target_deps is not None:
      cmake_info = info.get(CMakeDepsProvider)
      if cmake_info is not None:
        custom_target_deps.extend(cmake_info.targets)
    files_provider = info.get(FilesProvider)
    if files_provider is not None:
      return files_provider.paths
    raise ValueError(f"get_file_paths failed for {target} info {repr(info)}")

  def get_targets_file_paths(
      self,
      targets: Iterable[TargetId],
      custom_target_deps: Optional[List[CMakeTarget]] = None,
  ) -> List[str]:
    files = []
    for target in targets:
      files.extend(self.get_file_paths(target, custom_target_deps))
    return sorted(set(files))

  def generate_cmake_target_pair(self,
                                 target_id: TargetId,
                                 alias: bool = True) -> CMakeTargetPair:
    assert isinstance(target_id,
                      TargetId), f"Requires TargetId: {repr(target_id)}"

    cmake_target = self._cmake_dep_pairs.get(target_id)
    if cmake_target is not None:
      return cmake_target

    # If we're making a cmake name for a persisted target, assume that the
    # persisted target name is the alias.
    cmake_target_pair = self.workspace._persisted_canonical_name.get(target_id)
    if cmake_target_pair is not None:
      return cmake_target_pair

    cmake_package = self.workspace.get_cmake_package_name(
        target_id.repository_id)
    if cmake_package is None:
      raise ValueError(f"Unknown repo in target {target_id.as_label()}")
    pair = label_to_generated_cmake_target(target_id, cmake_package)
    if not alias:
      pair = pair.with_alias(None)
    return pair

  def add_required_dep_package(self, package: str) -> None:
    self.required_dep_packages.add(package)

  def _maybe_add_package_deps(self, p: Optional[CMakePackageDepsProvider]):
    if p is not None:
      for package in p.packages:
        if package not in self.workspace.repo_cmake_packages:
          assert not isinstance(package, bool)
          self.add_required_dep_package(package)

  def get_dep(self,
              target_id: TargetId,
              alias: bool = True) -> List[CMakeTarget]:
    """Maps a Bazel target to the corresponding CMake target."""
    # Local target.
    assert isinstance(target_id,
                      TargetId), f"Requires TargetId: {repr(target_id)}"
    info = self.get_optional_target_info(target_id)
    if info is not None:
      self._maybe_add_package_deps(info.get(CMakePackageDepsProvider))
      if info.get(CMakeDepsProvider):
        return info[CMakeDepsProvider].targets
      elif info.get(CMakeTargetPairProvider):
        print(info)
        return [info[CMakeTargetPairProvider].dep]

    # New untracked target.
    cmake_target = self.generate_cmake_target_pair(target_id, alias)
    if target_id not in self._cmake_dep_pairs:
      self.add_required_dep_package(cmake_target.cmake_package)
      # If it's not persisted, track it now.
      if target_id not in self.workspace._persisted_canonical_name:
        self._cmake_dep_pairs[target_id] = cmake_target
    return [cmake_target.dep]

  def get_deps(self, targets: List[TargetId]) -> List[CMakeTarget]:
    deps: List[CMakeTarget] = []
    for target in targets:
      deps.extend(self.get_dep(target))
    return deps

  def get_dummy_source(self):
    """Returns the path to a dummy source file.

    This is used in cases where at least one source file must be specified for a
    CMake target but there are none in the corresponding Bazel target.
    """
    dummy_source_relative_path = "bazel_to_cmake_empty_source.cc"
    dummy_source_path = str(
        pathlib.PurePosixPath(
            self.repo.cmake_binary_dir).joinpath(dummy_source_relative_path))
    if not self._wrote_dummy_source and not os.path.exists(dummy_source_path):
      pathlib.Path(dummy_source_path).write_bytes(b"")
    return dummy_source_path

  def load_library(self, target_id: TargetId) -> Dict[str, Any]:
    """Returns the global scope for the given bzl library target.

    Loads it if not already loaded in the current phase.  Libraries are loaded
    separately for the workspace loading and package loading phases.

    1. If the target has been overriden by a call to `register_bzl_library` or
       has been added as an ignored library to the workspace, the overridden
       implementation will be used.

    2. If the repository has been loaded (i.e. the current repo or the top-level
       repo if this is a dependency), then load the actual bzl file.

    3. Otherwise, ignore the library, i.e. return `IgnoredLibrary()`.
    """
    is_workspace = (self._phase == Phase.LOADING_WORKSPACE)
    key = (target_id, is_workspace)
    library = self._loaded_libraries.get(key)
    if library is not None:
      return library

    if target_id in self.workspace.ignored_libraries:
      # Specifically ignored.
      if self._verbose:
        print(f"Ignoring library: {target_id.as_label()}")
      library = IgnoredLibrary()
      self._loaded_libraries[key] = library
      return library

    library_type = get_bazel_library(key)
    if library_type is not None:
      library = library_type(self._evaluation_context, target_id, "builtin")
      self._loaded_libraries[key] = library
      return library

    if target_id.repository_id not in self.workspace.repos:
      # Unknown repository, ignore.
      if self._verbose:
        print(f"Unknown library: {target_id.as_label()}")
      return IgnoredLibrary()

    library_path = self.get_source_file_path(target_id)
    assert library_path is not None
    if self._verbose:
      print(f"Using library: {target_id.as_label()} at {library_path}")

    scope_type = BazelWorkspaceGlobals if is_workspace else BuildFileLibraryGlobals
    library = scope_type(self._evaluation_context, target_id, library_path)
    # Switch packages; A loaded library becomes the caller package_id as it
    # is evaluated.
    old_package = (self._evaluation_context.caller_package,
                   self._evaluation_context.caller_package_id)
    self._evaluation_context.update_current_package(
        package_id=target_id.package_id)

    try:
      # Load the library content.
      content = self._load(library_path)
      # Parse and evaluate the starlark script as a library.
      exec(compile(content, library_path, "exec"), library)  # pylint: disable=exec-used
    except Exception as e:
      raise RuntimeError(
          f"While loading {target_id.as_label()} ({library_path})") from e

    # Restore packages and save the library
    self._evaluation_context.update_current_package(*old_package)
    self._loaded_libraries[key] = library
    return library

  def _load(self, path: str) -> str:
    if self._verbose:
      print(f"Loading {path}")
    self.loaded_files.add(path)
    return pathlib.Path(path).read_text(encoding="utf-8")

  def process_build_file(self, build_file: str):
    """Processes a single package (BUILD file)."""
    build_file_path = str(
        pathlib.PurePosixPath(self.repo.source_directory).joinpath(build_file))
    self.process_build_content(build_file_path, self._load(build_file_path))

  def process_build_content(self, build_file_path: str, content: str):
    """Processes a single package (BUILD file content)."""
    assert build_file_path.startswith(self.repo.source_directory)
    self._phase = Phase.LOADING_BUILD

    # remove prefix and BUILD.
    package_name = build_file_path[(
        1 + len(self.repo.source_directory)):build_file_path.rfind("/")]

    package = Package(self.repo, package_name)
    build_target = package.package_id.get_target_id(
        os.path.basename(build_file_path))
    self._evaluation_context.update_current_package(package=package)

    scope = BuildFileGlobals(self._evaluation_context, build_target,
                             build_file_path)
    try:
      exec(compile(content, build_file_path, "exec"), scope)  # pylint: disable=exec-used
    except Exception as e:
      raise RuntimeError(
          f"While processing {repr(package.package_id)} ({build_file_path})"
      ) from e

  def process_workspace(self):
    """Processes the WORKSPACE."""
    assert self.repo.top_level
    workspace_file_path = str(
        pathlib.PurePosixPath(
            self.repo.source_directory).joinpath("WORKSPACE.bazel"))
    if not os.path.exists(workspace_file_path):
      workspace_file_path = str(
          pathlib.PurePosixPath(
              self.repo.source_directory).joinpath("WORKSPACE"))
    self.process_workspace_content(workspace_file_path,
                                   self._load(workspace_file_path))

  def process_workspace_content(self, workspace_file_path: str, content: str):
    assert (workspace_file_path.endswith("WORKSPACE") or
            workspace_file_path.endswith("WORKSPACE.bazel"))
    assert self.repo.top_level
    self._phase = Phase.LOADING_WORKSPACE

    workspace_target_id = self.repo.repository_id.get_package_id(
        "").get_target_id(os.path.basename(workspace_file_path))

    self._evaluation_context.update_current_package(
        package_id=workspace_target_id.package_id)

    scope = BazelWorkspaceGlobals(self._evaluation_context, workspace_target_id,
                                  workspace_file_path)

    exec(compile(content, workspace_file_path, "exec"), scope)  # pylint: disable=exec-used
    for callback in self._call_after_workspace_loading:
      callback()

  def call_after_workspace_loading(self, callback: Callable[[], None]) -> None:
    assert self._phase == Phase.LOADING_WORKSPACE
    self._call_after_workspace_loading.append(callback)

  def call_after_analysis(self, callback: Callable[[], None]) -> None:
    self._call_after_analysis.append(callback)


class EvaluationContext(InvocationContext):
  """Implements InvocationContext interface for EvaluationState."""

  __slots__ = ("_state", "_caller_package_id", "_caller_package")

  def __init__(self,
               state: EvaluationState,
               package_id: PackageId,
               package: Optional[Package] = None):
    assert state
    self._state = state
    self._caller_package_id = package_id
    self._caller_package = package
    assert self._caller_package_id

  def __repr__(self):
    d = {k: getattr(self, k) for k in self.__slots__}
    return f"<{self.__class__.__name__}>: {d}"

  def update_current_package(self,
                             package: Optional[Package] = None,
                             package_id: Optional[PackageId] = None) -> None:
    if package_id is None:
      assert package is not None
      package_id = package.package_id
    self._caller_package_id = package_id
    self._caller_package = package
    assert self._caller_package_id

  def _get_repository(self, repository_id: RepositoryId) -> Repository:
    repo = self._state.workspace.repos.get(repository_id)
    if repo is None:
      raise ValueError(f"Unknown bazel repo: {repository_id}")
    return repo

  # Derived fields
  def snapshot(self) -> "EvaluationContext":
    return copy.copy(self)

  @property
  def caller_package(self) -> Optional[Package]:
    return self._caller_package

  @property
  def caller_package_id(self) -> PackageId:
    return self._caller_package_id

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
    return super().access(provider_type)

  def resolve_source_root(self, repository_id: RepositoryId) -> str:
    return self._get_repository(repository_id).source_directory

  def resolve_output_root(self, repository_id: RepositoryId) -> str:
    return self._get_repository(repository_id).cmake_binary_dir

  def resolve_repo_mapping(
      self, target_id: TargetId,
      mapping_repository_id: Optional[RepositoryId]) -> TargetId:
    # Resolve repository mappings
    if mapping_repository_id is None:
      assert self._caller_package_id
      mapping_repository_id = self._caller_package_id.repository_id
    target = remap_target_repo(
        target_id,
        self._get_repository(mapping_repository_id).repo_mapping)
    # Resolve bindings.
    if target.package_name == "external":
      repo = self._get_repository(target.repository_id)
      while target in repo.bindings:
        target = repo.bindings[target]
    return target

  def load_library(self, target: TargetId) -> Dict[str, Any]:
    return self._state.load_library(target)

  def get_target_info(self, target_id: TargetId) -> TargetInfo:
    return self._state.get_target_info(target_id)

  def add_rule(self,
               rule_id: TargetId,
               impl: RuleImpl,
               outs: Optional[List[TargetId]] = None,
               visibility: Optional[List[RelativeLabel]] = None,
               **kwargs) -> None:
    if visibility is not None:
      if kwargs.get("analyze_by_default") is None:
        assert self._caller_package
        kwargs["analyze_by_default"] = Visibility(
            self._caller_package).analyze_by_default(
                self.resolve_target_or_label_list(visibility))
    self._state.add_rule(rule_id, impl, outs, **kwargs)

  def add_analyzed_target(self, target_id: TargetId, info: TargetInfo) -> None:
    self._state.add_analyzed_target(target_id, info)
