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

import collections
from collections.abc import Iterable
import enum
import json
import os
import pathlib
from typing import Any, Callable, NamedTuple, Tuple, TypeVar

from . import cmake_builder
from .active_repository import Repository
from .bzl_library.register import get_bzl_library
from .cmake_builder import CMakeBuilder
from .cmake_provider import CMakeExecutableTargetProvider
from .cmake_provider import CMakeHallucinatedTarget
from .cmake_provider import CMakeLinkLibrariesProvider
from .cmake_provider import CMakePackageDepsProvider
from .cmake_target import CMakePackage
from .cmake_target import CMakeTargetPair
from .evaluation_context import EvaluationContext
from .evaluation_state import EvaluationState
from .module_resolution import ModuleResolver
from .ordered_set import OrderedSet
from .package import Package
from .provider_util import ProviderCollection
from .starlark.bazel_target import TargetId
from .starlark.common_providers import BuildSettingInfo
from .starlark.common_providers import ConditionProvider
from .starlark.common_providers import FilesProvider
from .starlark.exec import compile_and_exec
from .starlark.ignored import IgnoredLibrary
from .starlark.label import RelativeLabel
from .starlark.provider import TargetInfo
from .starlark.scope_build_file import ScopeBuildBzlFile
from .starlark.scope_build_file import ScopeBuildFile
from .starlark.scope_module_file import ScopeModuleFile
from .starlark.scope_workspace_file import ScopeWorkspaceFile
from .starlark.select import Configurable
from .starlark.select import Select
from .starlark.select import SelectExpression
from .util import cmake_is_true
from .util import is_relative_to
from .workspace import Workspace

T = TypeVar("T")

RuleImpl = Callable[[], None]


class RuleInfo(NamedTuple):
  mnemonic: str  # Only used for debugging currently.
  callers: list[str]
  outs: list[TargetId]
  impl: RuleImpl


class Phase(enum.Enum):
  LOADING_WORKSPACE = 1
  LOADING_MODULE = 2
  LOADING_BUILD = 3
  ANALYZE = 4


class EvaluationImpl(EvaluationState):
  """Implementation of EvaluationState."""

  def __init__(self, active_repo: Repository):
    self.active_repo: Repository = active_repo
    self.builder = CMakeBuilder()
    self._evaluation_context = EvaluationContext(
        self, self.active_repo.repository_id.get_package_id("")
    )
    self.public_only = not (
        self.active_repo.top_level
        and cmake_is_true(self.workspace.cmake_vars["PROJECT_IS_TOP_LEVEL"])
    )
    self.loaded_files: OrderedSet[str] = OrderedSet()
    self._loaded_libraries: dict[Tuple[TargetId, bool], dict[str, Any]] = dict()
    self._wrote_placeholder_source = False
    self.errors: list[str] = []
    # Track CMakePackage dependencies.
    self._required_dep_packages: OrderedSet[CMakePackage] = OrderedSet()
    # Track CMakeTargetPair dependencies
    self._required_dep_targets: dict[TargetId, CMakeTargetPair] = {}
    # Maps targets to their rules.
    self._unanalyzed_rules: OrderedSet[TargetId] = OrderedSet()
    self._all_rules: dict[TargetId, RuleInfo] = {}
    self._unanalyzed_targets: dict[TargetId, TargetId] = {}
    self._targets_to_analyze: OrderedSet[TargetId] = OrderedSet()
    self._analyzed_targets: dict[TargetId, TargetInfo] = {}
    self._call_after_workspace_loading = collections.deque()
    self._call_after_analysis = collections.deque()
    self._stack = collections.deque()
    self._phase: Phase = Phase.LOADING_WORKSPACE
    self.module_resolver = ModuleResolver(self)

  def __repr__(self):
    return (
        f"<{self.__class__.__name__}>: "
        "{\n"
        f"  _stack: {repr(self._stack)},\n"
        f"  _repo: {repr(self.active_repo.repository)},\n"
        "}\n"
    )

  @property
  def workspace(self) -> Workspace:
    return self.active_repo.workspace

  @property
  def verbose(self) -> int:
    return self.active_repo.workspace.verbose

  @property
  def evaluation_context(self) -> EvaluationContext:
    return self._evaluation_context

  @property
  def targets_to_analyze(self) -> list[TargetId]:
    return sorted(self._targets_to_analyze)

  def resolve_target(self, target: str) -> TargetId:
    return self._evaluation_context.resolve_target(target)

  def analyze(self, targets: list[TargetId]):
    """Analayze the transitive dependencies of `targets`.

    Note that any targets that are skipped will not be available for use as
    dependencies of targets defined in other repositories.
    """
    self._phase = Phase.ANALYZE
    for target in targets:
      if self.verbose:
        print(f"Analyze: {target.as_label()}")
      assert isinstance(target, TargetId)
      self.get_target_info(target)

    for package in sorted(self._required_dep_packages):
      self.builder.find_package(
          package, section=cmake_builder.FIND_DEP_PACKAGE_SECTION
      )

    if self.verbose > 1 and self._call_after_analysis:
      print(f"call_after_analysis: {len(self._call_after_analysis)}")
    while self._call_after_analysis:
      callback = self._call_after_analysis.pop()
      callback()

  def add_rule_impl(
      self,
      _mnemonic: str,
      _callers: list[str],
      *,
      rule_id: TargetId,
      impl: RuleImpl,
      outs: list[TargetId] | None = None,
      analyze_by_default: bool = True,
  ) -> None:
    """Adds a rule.

    Args:
      rule_label: Label for the rule.
      impl: Implementation function called during analysis phase.
      outs: Additional output targets beyond `rule_label` itself.
      analyze_by_default: Whether to analyze by default, as opposed to only if
        it is a dependency of another target being analyzed.
    """
    assert isinstance(rule_id, TargetId), f"Requires TargetId: {repr(rule_id)}"
    # kind is assigned from caller function name
    if outs is None:
      outs = []
    r = RuleInfo(_mnemonic, _callers, outs=outs, impl=impl)
    if rule_id in self._all_rules:
      raise ValueError(
          f"Duplicate rule: {rule_id.as_label()} from {r}\n"
          f"Existing rule: {self._all_rules[rule_id]}"
      )
    self._all_rules[rule_id] = r
    self._unanalyzed_rules.add(rule_id)
    self._unanalyzed_targets[rule_id] = rule_id
    for out_id in r.outs:
      if out_id in self._unanalyzed_targets or out_id in self._analyzed_targets:
        detail = ""
        if out_id in self._all_rules:
          detail = f"\nExisting rule: {self._all_rules[out_id]}"
        raise ValueError(
            f"Duplicate output: {out_id.as_label()} from {r}{detail}"
        )
      self._unanalyzed_targets[out_id] = rule_id
    if analyze_by_default:
      self._targets_to_analyze.add(rule_id)
    if self.verbose:
      print(f"add_rule: {rule_id.as_label()} as {r}")

  @property
  def required_dep_packages(self) -> Iterable[str]:
    return self._required_dep_packages

  def add_analyzed_target(self, target_id: TargetId, info: TargetInfo) -> None:
    """Adds the `TargetInfo' for an analyzed target.

    This must be called by the `RuleImpl` function for the rule_label and each
    output target.
    """
    assert isinstance(
        target_id, TargetId
    ), f"Requires TargetId: {repr(target_id)}"
    assert info is not None
    self._analyzed_targets[target_id] = info

    if (
        info.get(CMakeExecutableTargetProvider) is not None
        or info.get(CMakeLinkLibrariesProvider) is not None
    ):
      self._required_dep_targets.pop(target_id, None)
    if (self.verbose > 1) or (
        self.verbose and info.get(FilesProvider) is not None
    ):
      print(f"add_analyzed_target: {target_id.as_label()} with {info}")

  def visit_analyzed_targets(
      self, visitor: Callable[[TargetId, TargetInfo], None]
  ):
    for target, info in self._analyzed_targets.items():
      visitor(target, info)

  def visit_required_dep_targets(
      self, visitor: Callable[[TargetId, CMakeTargetPair], None]
  ):
    for target, target_pair in self._required_dep_targets.items():
      visitor(target, target_pair)

  def _get_target_info(
      self, target_id: TargetId, optional: bool
  ) -> TargetInfo | None:
    assert isinstance(
        target_id, TargetId
    ), f"Requires TargetId: {repr(target_id)}"
    info = self._analyzed_targets.get(target_id, None)
    if info is not None:
      return info

    rule_id = self._unanalyzed_targets.get(target_id, None)
    if rule_id is None:
      # Is this a global persistent target?
      info = self.workspace._persisted_target_info.get(target_id)
      if info is not None:
        return info

      # Is this a source file?
      source_path = self.get_source_file_path(target_id)
      if source_path and os.path.isfile(source_path):
        info = TargetInfo(FilesProvider([source_path.as_posix()]))
        self.add_analyzed_target(target_id, info)
        return info

      if optional:
        return None
      if source_path is None:
        raise ValueError(
            f"Error analyzing {target_id.as_label()}: Unknown repository\n"
            f"{self._all_rules.get(target_id, None)}"
        )
      raise ValueError(
          f"Error analyzing {target_id.as_label()}: File not found "
          f"{source_path}\n"
          f"{self._all_rules.get(target_id, None)}"
      )

    # At this point a rule_info instance is expected.
    rule_info = self._all_rules.get(rule_id, None)
    if rule_info is None:
      raise ValueError(
          f"Error analyzing {target_id.as_label()}: No rule found for"
          f" {rule_id.as_label()}"
      )

    # Mark the rule as analyzed.
    if rule_id not in self._unanalyzed_rules:
      raise ValueError(
          f"Error analyzing {rule_id.as_label()}: Already analyzed?"
      )
    self._unanalyzed_rules.remove(rule_id)
    self._unanalyzed_targets.pop(rule_id, None)

    try:
      self._stack.append((target_id, rule_info))
      if self.verbose > 2:
        print(f"Invoking {target_id.as_label()}: {repr(rule_info)}")
      rule_info.impl()
    except Exception as e:
      e.args = (e.args if e.args else tuple()) + (
          f"collecting target {target_id.as_label()}",
      )
      raise
    finally:
      self._stack.pop()

    for out_id in rule_info.outs:
      if out_id not in self._analyzed_targets:
        raise ValueError(
            f"Error analyzing {rule_id.as_label()} with {rule_info}: Expected"
            f" add_analyzed_target({out_id.as_label()})"
        )
      self._unanalyzed_targets.pop(out_id, None)

    info = self._analyzed_targets.get(target_id, None)
    if info is None and not optional:
      raise ValueError(
          f"Error analyzing {rule_id.as_label()}: {rule_info} produced no"
          " analyzed target"
      )
    return info

  def get_optional_target_info(self, target_id: TargetId) -> TargetInfo | None:
    return self._get_target_info(target_id, True)

  def get_target_info(self, target_id: TargetId) -> TargetInfo:
    info = self._get_target_info(target_id, False)
    assert info is not None
    return info

  def get_source_file_path(
      self, target_id: TargetId
  ) -> pathlib.PurePath | None:
    assert isinstance(target_id, TargetId)
    repo = self.workspace.all_repositories.get(target_id.repository_id)
    if repo is None:
      return None
    return repo.get_source_file_path(target_id)

  def get_generated_file_path(self, target_id: TargetId) -> pathlib.PurePath:
    assert isinstance(target_id, TargetId)
    repo = self.workspace.all_repositories.get(target_id.repository_id)
    if repo is None:
      raise ValueError(
          f"Unknown repository name in target: {target_id.as_label()}"
      )
    return repo.get_generated_file_path(target_id)

  def evaluate_condition(self, target_id: TargetId) -> Any:
    assert isinstance(target_id, TargetId)
    assert self._phase == Phase.ANALYZE

    target_info = self.get_target_info(target_id)
    if target_info.get(ConditionProvider) is not None:
      return target_info[ConditionProvider].value
    if target_info.get(BuildSettingInfo) is not None:
      return target_info[BuildSettingInfo].value

    print(f"Using {target_id.as_label()} as false condition.")
    return False

  def evaluate_configurable(self, configurable: Configurable[T]) -> T:
    """Evaluates a `Configurable` expression."""
    assert self._phase == Phase.ANALYZE
    if isinstance(configurable, Select) or isinstance(
        configurable, SelectExpression
    ):
      return configurable.evaluate(self.evaluate_condition)
    return configurable

  def collect_targets(
      self,
      targets: Iterable[TargetId],
      collector: ProviderCollection | None = None,
  ) -> ProviderCollection:
    if collector is None:
      collector = ProviderCollection()
    for t in targets:
      assert isinstance(t, TargetId), f"Requires TargetId: {repr(t)}"
      try:
        target_info = self.get_target_info(t)
        collector.collect(t, target_info)
        assert (
            target_info.get(FilesProvider) is not None
        ), f"No FilesProvider for target: {t.as_label()} {target_info}"
      except Exception as e:
        e.args = (e.args if e.args else tuple()) + (
            f"collecting target {t.as_label()}",
        )
        raise
    return collector

  def collect_deps(
      self,
      targets: Iterable[TargetId],
      collector: ProviderCollection | None = None,
      alias: bool = True,
  ) -> ProviderCollection:
    if collector is None:
      collector = ProviderCollection()
    for t in targets:
      try:
        collector.collect(t, self._get_dep(t, alias))
      except Exception as e:
        e.args = (e.args if e.args else tuple()) + (
            f"collecting target {t.as_label()}",
        )
        raise
    return collector

  def _generate_cmake_target_pair_imp(
      self, target_id: TargetId, alias: bool = True
  ) -> CMakeTargetPair:
    repo = self.workspace.all_repositories.get(target_id.repository_id)
    if repo is None:
      raise ValueError(
          f"Unknown repository {target_id.repository_id} in target"
          f" {target_id.as_label()}"
      )
    cmake_target_pair = repo.get_cmake_target_pair(target_id)
    if alias:
      return cmake_target_pair
    return cmake_target_pair.with_alias(None)

  def generate_cmake_target_pair(
      self, target_id: TargetId, alias: bool = True
  ) -> CMakeTargetPair:
    assert isinstance(
        target_id, TargetId
    ), f"Requires TargetId: {repr(target_id)}"
    # If this package is already a required dependency, return that.
    cmake_target = self._required_dep_targets.get(target_id)
    if cmake_target is not None:
      return cmake_target
    # Generate a new name
    return self._generate_cmake_target_pair_imp(target_id, alias)

  def add_required_dep_package(self, package: CMakePackage) -> None:
    if package != self.active_repo.cmake_project_name:
      self._required_dep_packages.add(package)

  def _get_dep(
      self,
      target_id: TargetId,
      alias: bool,
  ) -> TargetInfo:
    """Maps a Bazel target to the corresponding CMake target."""
    # Local target.
    assert isinstance(
        target_id, TargetId
    ), f"Requires TargetId: {repr(target_id)}"
    info = self.get_optional_target_info(target_id)
    if info is not None:
      if info.get(CMakePackageDepsProvider):
        self.add_required_dep_package(info[CMakePackageDepsProvider].package)
      return info

    # If this package is already a required dependency, return that.
    cmake_target = self._required_dep_targets.get(target_id)
    if cmake_target is None:
      # Generate a new name.
      cmake_target = self._generate_cmake_target_pair_imp(target_id, alias)
      self._required_dep_targets[target_id] = cmake_target

    self.add_required_dep_package(cmake_target.cmake_package)
    return TargetInfo(CMakeHallucinatedTarget(cmake_target.dep))

  def get_placeholder_source(self):
    """Returns the path to a placeholder source file.

    This is used in cases where at least one source file must be specified for a
    CMake target but there are none in the corresponding Bazel target.
    """
    placeholder_source_relative_path = "bazel_to_cmake_empty_source.cc"
    placeholder_source_path = self.active_repo.cmake_binary_dir.joinpath(
        placeholder_source_relative_path
    )
    if not self._wrote_placeholder_source and not os.path.exists(
        placeholder_source_path.as_posix()
    ):
      pathlib.Path(placeholder_source_path).write_bytes(b"")
    return placeholder_source_path.as_posix()

  def load_library(self, target_id: TargetId) -> dict[str, Any]:
    """Returns the global scope for the given bzl library target.

    Loads it if not already loaded in the current phase.  Libraries are loaded
    separately for the workspace loading and package loading phases.

    1. If the target has been overridden by a call to `register_bzl_library` or
       has been added as an ignored library to the workspace, the overridden
       implementation will be used.

    2. If the repository has been loaded (i.e. the current repo or the top-level
       repo if this is a dependency), then load the actual bzl file.

    3. Otherwise, ignore the library, i.e. return `IgnoredLibrary()`.
    """
    is_workspace = self._phase == Phase.LOADING_WORKSPACE
    key = (target_id, is_workspace)
    library = self._loaded_libraries.get(key)
    if library is not None:
      return library

    if target_id in self.active_repo.ignored_libraries:
      # Specifically ignored.
      if self.verbose:
        print(f"Ignoring library: {target_id.as_label()}")
      library = IgnoredLibrary()
      self._loaded_libraries[key] = library
      return library

    library_type = get_bzl_library(key)
    if library_type is not None:
      if self.verbose:
        print(f"Builtin library: {target_id.as_label()}")
      library = library_type(self._evaluation_context, target_id, "builtin")
      self._loaded_libraries[key] = library
      return library

    library_path = self.get_source_file_path(target_id)
    if library_path is None:
      print(f"Unknown library: {target_id.as_label()}")
      return IgnoredLibrary()

    if not pathlib.Path(library_path).exists():
      print(f"Missing library: {target_id.as_label()} at {library_path}")
      return IgnoredLibrary()

    if self.verbose:
      print(f"Loading library: {target_id.as_label()} at {library_path}")

    scope_type = ScopeWorkspaceFile if is_workspace else ScopeBuildBzlFile
    library = scope_type(
        self._evaluation_context, target_id, library_path.as_posix()
    )
    # Switch packages; A loaded library becomes the caller package_id as it
    # is evaluated.
    self._stack.append((
        self._evaluation_context.caller_package,
        self._evaluation_context.caller_package_id,
    ))
    self._evaluation_context.update_current_package(
        package_id=target_id.package_id
    )
    self.loaded_files.add(library_path.as_posix())

    # Load, parse and evaluate the starlark script as a library.
    compile_and_exec(
        None,
        library_path.as_posix(),
        library,
        f" loading library {target_id.as_label()}",
    )
    library.__target__ = target_id.as_label()

    # Restore packages and save the library
    self._evaluation_context.update_current_package(*self._stack.pop())
    self._loaded_libraries[key] = library
    return library

  def process_build_file(self, build_file: pathlib.PurePath):
    """Processes a single package (BUILD file)."""
    build_file_path = self.active_repo.source_directory.joinpath(build_file)
    if self.verbose:
      print(f"Loading {build_file_path}")
    self.loaded_files.add(build_file_path.as_posix())
    self.process_build_content(
        build_file_path,
        pathlib.Path(build_file_path).read_text(encoding="utf-8"),
    )

  def process_build_content(
      self, build_file_path: pathlib.PurePath, content: str
  ):
    """Processes a single package (BUILD file content)."""
    if isinstance(build_file_path, pathlib.PureWindowsPath):
      build_file_path = pathlib.PurePath(build_file_path.as_posix())

    assert is_relative_to(build_file_path, self.active_repo.source_directory)

    # remove prefix and BUILD.
    package_path = build_file_path.relative_to(
        self.active_repo.source_directory
    ).parent
    package_name = (
        package_path.as_posix() if package_path != pathlib.PurePath() else ""
    )

    self._phase = Phase.LOADING_BUILD

    package = Package(self.active_repo, package_name)
    build_target = package.package_id.get_target_id(build_file_path.name)
    self._evaluation_context.update_current_package(package=package)

    scope = ScopeBuildFile(
        self._evaluation_context, build_target, build_file_path.as_posix()
    )

    # Load, parse and evaluate the starlark script as a library.
    compile_and_exec(
        content,
        build_file_path.as_posix(),
        scope,
        f" processing {repr(package.package_id)}",
    )

  def process_workspace(self) -> bool:
    """Processes the WORKSPACE."""
    assert self.active_repo.top_level
    workspace_file_path = self.active_repo.source_directory.joinpath(
        "WORKSPACE.bazel"
    )
    if not pathlib.Path(workspace_file_path).exists():
      workspace_file_path = self.active_repo.source_directory.joinpath(
          "WORKSPACE"
      )

    if not pathlib.Path(workspace_file_path).exists():
      return False
    if self.verbose:
      print(f"Loading {workspace_file_path}")
    self.loaded_files.add(workspace_file_path.as_posix())
    self.process_workspace_content(
        workspace_file_path,
        pathlib.Path(workspace_file_path).read_text(encoding="utf-8"),
    )
    return True

  def process_workspace_content(
      self, workspace_file_path: pathlib.PurePath, content: str
  ):
    if isinstance(workspace_file_path, pathlib.PureWindowsPath):
      workspace_file_path = pathlib.PurePath(workspace_file_path.as_posix())

    assert self.active_repo.top_level

    workspace_target_id = self.active_repo.repository_id.get_package_id(
        ""
    ).get_target_id(workspace_file_path.name)

    self._phase = Phase.LOADING_WORKSPACE

    self._evaluation_context.update_current_package(
        package_id=workspace_target_id.package_id
    )
    scope = ScopeWorkspaceFile(
        self._evaluation_context,
        workspace_target_id,
        workspace_file_path.as_posix(),
    )

    compile_and_exec(
        content,
        workspace_file_path.as_posix(),
        scope,
    )
    while self._call_after_workspace_loading:
      callback = self._call_after_workspace_loading.pop()
      callback()

  def process_module(self) -> bool:
    """Processes the MODULE.bazel."""
    assert self.active_repo.top_level
    module_file_path = self.active_repo.source_directory.joinpath(
        "MODULE.bazel"
    )
    lockfile_path = self.active_repo.source_directory.joinpath(
        "MODULE.bazel.lock"
    )

    if not pathlib.Path(module_file_path).exists():
      return False

    if self.verbose:
      print(f"Loading {module_file_path}")

    # First, process the lockfile.
    if pathlib.Path(lockfile_path).exists():
      self.process_module_lockfile(lockfile_path)

    # Then, process the module file itself.
    self.loaded_files.add(module_file_path.as_posix())
    self.process_module_content(
        module_file_path,
        pathlib.Path(module_file_path).read_text(encoding="utf-8"),
    )
    return True

  def process_module_content(
      self, module_file_path: pathlib.PurePath, content: str
  ):
    """Processes the MODULE.bazel content."""
    if isinstance(module_file_path, pathlib.PureWindowsPath):
      module_file_path = pathlib.PurePath(module_file_path.as_posix())

    assert self.active_repo.top_level

    module_target_id = self.active_repo.repository_id.get_package_id(
        ""
    ).get_target_id(module_file_path.name)

    self._phase = Phase.LOADING_MODULE
    self._evaluation_context.update_current_package(
        package_id=module_target_id.package_id
    )
    scope = ScopeModuleFile(
        self._evaluation_context,
        module_target_id,
        module_file_path.as_posix(),
    )
    compile_and_exec(
        content,
        module_file_path.as_posix(),
        scope,
    )

  def process_module_lockfile(self, lockfile_path: pathlib.Path):
    """Processes the MODULE.bazel.lock file."""
    if self.verbose:
      print(f"Loading lockfile {lockfile_path}")
    try:
      with open(lockfile_path, "r", encoding="utf-8") as f:
        lockfile_json = json.load(f)
        self.module_resolver.load_lockfile(lockfile_json)
    except Exception as e:
      print(f"Warning: Failed to load lockfile {lockfile_path}: {e}")

  def set_module_info(
      self, name: str, version: str, compatibility_level: int, repo_name: str
  ) -> None:
    self.module_resolver.set_module_name_version(name, version)

  def add_bazel_dep(
      self,
      name: str,
      version: str,
      max_compatibility_level: int,
      repo_name: str,
      dev_dependency: bool,
  ) -> None:
    self.module_resolver.add_bazel_dep(
        name, version, max_compatibility_level, repo_name, dev_dependency
    )

  def use_repo(self, extension_proxy, *args, **kwargs) -> None:
    self.module_resolver.use_repo(extension_proxy, *args, **kwargs)

  def add_module_override(
      self, module_name: str, override_info: dict[str, Any]
  ) -> None:
    self.module_resolver.add_module_override(module_name, override_info)

  def include_module_file(self, label: RelativeLabel, scope: Any) -> None:
    self.module_resolver.include_module_file(label, scope)

  def call_after_workspace_loading(self, callback: Callable[[], None]) -> None:
    assert self._phase == Phase.LOADING_WORKSPACE
    self._call_after_workspace_loading.appendleft(callback)

  def call_after_analysis(self, callback: Callable[[], None]) -> None:
    self._call_after_analysis.appendleft(callback)
