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

import ast
import collections
import copy
import enum
import functools
import inspect
import os
import pathlib
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple, Type, TypeVar, cast

from . import cmake_builder
from .cmake_builder import CMakeBuilder
from .cmake_target import CMakeDepsProvider
from .cmake_target import CMakeExecutableTargetProvider
from .cmake_target import CMakeLibraryTargetProvider
from .cmake_target import CMakePackage
from .cmake_target import CMakePackageDepsProvider
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetPair
from .package import Package
from .package import Visibility
from .starlark import dict_polyfill
from .starlark.bazel_globals import BazelWorkspaceGlobals
from .starlark.bazel_globals import BuildFileGlobals
from .starlark.bazel_globals import BuildFileLibraryGlobals
from .starlark.bazel_globals import get_bazel_library
from .starlark.bazel_target import PackageId
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
from .util import is_relative_to
from .workspace import Repository
from .workspace import Workspace

T = TypeVar("T")

RuleImpl = Callable[[], None]


class RuleInfo(NamedTuple):
  mnemonic: str  # Only used for debugging currently.
  callers: List[str]
  outs: List[TargetId]
  impl: RuleImpl


class Phase(enum.Enum):
  LOADING_WORKSPACE = 1
  LOADING_BUILD = 2
  ANALYZE = 3


class EvaluationState:
  """State used while evaluating Starlark code."""

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
    self.loaded_files: Set[str] = set()
    self._loaded_libraries: Dict[Tuple[TargetId, bool], Dict[str, Any]] = dict()
    self._wrote_placeholder_source = False
    self.errors: List[str] = []
    # Track CMakePackage dependencies.
    self._required_dep_packages: Set[CMakePackage] = set()
    # Track CMakeTargetPair dependencies
    self._required_dep_targets: Dict[TargetId, CMakeTargetPair] = {}
    # Maps targets to their rules.
    self._unanalyzed_rules: Set[TargetId] = set()
    self._all_rules: Dict[TargetId, RuleInfo] = {}
    self._unanalyzed_targets: Dict[TargetId, TargetId] = {}
    self._targets_to_analyze: Set[TargetId] = set()
    self._analyzed_targets: Dict[TargetId, TargetInfo] = {}
    self._call_after_workspace_loading = collections.deque()
    self._call_after_analysis = collections.deque()
    self._stack = collections.deque()
    self._phase: Phase = Phase.LOADING_WORKSPACE

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
  def targets_to_analyze(self) -> List[TargetId]:
    return sorted(self._targets_to_analyze)

  def analyze(self, targets: List[TargetId]):
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
      _callers: List[str],
      *,
      rule_id: TargetId,
      impl: RuleImpl,
      outs: Optional[List[TargetId]] = None,
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
    if rule_id in self._all_rules:
      raise ValueError(f"Duplicate rule: {rule_id.as_label()}")
    if outs is None:
      outs = []
    # kind is assigned from caller function name
    r = RuleInfo(_mnemonic, _callers, outs=outs, impl=impl)
    self._all_rules[rule_id] = r
    self._unanalyzed_rules.add(rule_id)
    self._unanalyzed_targets[rule_id] = rule_id
    for out_id in r.outs:
      if out_id in self._unanalyzed_targets or out_id in self._analyzed_targets:
        raise ValueError(f"Duplicate output: {out_id.as_label()}")
      self._unanalyzed_targets[out_id] = rule_id
    if analyze_by_default:
      self._targets_to_analyze.add(rule_id)
    if self.verbose:
      print(f"add_rule: {rule_id.as_label()} as {r}")

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
        or info.get(CMakeLibraryTargetProvider) is not None
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
  ) -> Optional[TargetInfo]:
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
      if source_path is None:
        if optional:
          return None
        raise ValueError(
            f"Error analyzing {target_id.as_label()}: Unknown repository"
        )
      if not os.path.isfile(source_path):
        if optional:
          return None
        raise ValueError(
            f"Error analyzing {target_id.as_label()}: File not found"
            f" {source_path}"
        )
      info = TargetInfo(FilesProvider([source_path.as_posix()]))
      self.add_analyzed_target(target_id, info)
      return info

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

    self._stack.append((target_id, rule_info))
    if self.verbose > 2:
      print(f"Invoking {target_id.as_label()}: {repr(rule_info)}")
    rule_info.impl()
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

  def get_optional_target_info(
      self, target_id: TargetId
  ) -> Optional[TargetInfo]:
    return self._get_target_info(target_id, True)

  def get_target_info(self, target_id: TargetId) -> TargetInfo:
    info = self._get_target_info(target_id, False)
    assert info is not None
    return info

  def get_source_file_path(
      self, target_id: TargetId
  ) -> Optional[pathlib.PurePath]:
    assert isinstance(target_id, TargetId)
    repo = self.workspace.all_repositories.get(target_id.repository_id)
    if repo is None:
      return None
    return repo.get_source_file_path(target_id)

  def get_generated_file_path(self, target_id: TargetId) -> pathlib.PurePath:
    repo = self.workspace.all_repositories.get(target_id.repository_id)
    if repo is None:
      raise ValueError(
          f"Unknown repository name in target: {target_id.as_label()}"
      )
    return repo.get_generated_file_path(target_id)

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
    if isinstance(configurable, Select) or isinstance(
        configurable, SelectExpression
    ):
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

  def _generate_cmake_target_pair_imp(
      self, target_id: TargetId, alias: bool = True
  ) -> CMakeTargetPair:
    repo = self.workspace.all_repositories.get(target_id.repository_id)
    if repo is None:
      raise ValueError(f"Unknown repo in target {target_id.as_label()}")
    cmake_target_pair = repo.get_cmake_target_pair(target_id)
    if not alias:
      cmake_target_pair = cmake_target_pair.with_alias(None)
    return cmake_target_pair

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

  def get_dep(
      self, target_id: TargetId, alias: bool = True
  ) -> List[CMakeTarget]:
    """Maps a Bazel target to the corresponding CMake target."""
    # Local target.
    assert isinstance(
        target_id, TargetId
    ), f"Requires TargetId: {repr(target_id)}"
    info = self.get_optional_target_info(target_id)
    if info is not None:
      if info.get(CMakePackageDepsProvider):
        for package in info[CMakePackageDepsProvider].packages:
          self.add_required_dep_package(package)
      if info.get(CMakeDepsProvider):
        return info[CMakeDepsProvider].targets.copy()

    # If this package is already a required dependency, return that.
    cmake_target = self._required_dep_targets.get(target_id)
    if cmake_target is None:
      # Generate a new name.
      cmake_target = self._generate_cmake_target_pair_imp(target_id, alias)
      self._required_dep_targets[target_id] = cmake_target

    self.add_required_dep_package(cmake_target.cmake_package)
    return [cmake_target.dep]

  def get_deps(self, targets: List[TargetId]) -> List[CMakeTarget]:
    deps: List[CMakeTarget] = []
    for target in targets:
      deps.extend(self.get_dep(target))
    return deps

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
    is_workspace = self._phase == Phase.LOADING_WORKSPACE
    key = (target_id, is_workspace)
    library = self._loaded_libraries.get(key)
    if library is not None:
      return library

    if target_id in self.workspace.ignored_libraries:
      # Specifically ignored.
      if self.verbose:
        print(f"Ignoring library: {target_id.as_label()}")
      library = IgnoredLibrary()
      self._loaded_libraries[key] = library
      return library

    library_type = get_bazel_library(key)
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

    scope_type = (
        BazelWorkspaceGlobals if is_workspace else BuildFileLibraryGlobals
    )
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

    try:
      # Load the library content.
      content = pathlib.Path(library_path).read_text(encoding="utf-8")
      # Parse and evaluate the starlark script as a library.
      _exec_module(content, library_path.as_posix(), library)
    except Exception as e:
      raise RuntimeError(
          f"While loading {target_id.as_label()} ({library_path})"
      ) from e

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

    scope = BuildFileGlobals(
        self._evaluation_context, build_target, build_file_path.as_posix()
    )
    try:
      _exec_module(content, build_file_path.as_posix(), scope)
    except Exception as e:
      raise RuntimeError(
          f"While processing {repr(package.package_id)} ({build_file_path})"
      ) from e

  def process_workspace(self):
    """Processes the WORKSPACE."""
    assert self.active_repo.top_level
    workspace_file_path = self.active_repo.source_directory.joinpath(
        "WORKSPACE.bazel"
    )
    if not pathlib.Path(workspace_file_path).exists():
      workspace_file_path = self.active_repo.source_directory.joinpath(
          "WORKSPACE"
      )

    if self.verbose:
      print(f"Loading {workspace_file_path}")
    self.loaded_files.add(workspace_file_path.as_posix())
    self.process_workspace_content(
        workspace_file_path,
        pathlib.Path(workspace_file_path).read_text(encoding="utf-8"),
    )

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
    scope = BazelWorkspaceGlobals(
        self._evaluation_context,
        workspace_target_id,
        workspace_file_path.as_posix(),
    )

    _exec_module(content, workspace_file_path.as_posix(), scope)
    while self._call_after_workspace_loading:
      callback = self._call_after_workspace_loading.pop()
      callback()

  def call_after_workspace_loading(self, callback: Callable[[], None]) -> None:
    assert self._phase == Phase.LOADING_WORKSPACE
    self._call_after_workspace_loading.appendleft(callback)

  def call_after_analysis(self, callback: Callable[[], None]) -> None:
    self._call_after_analysis.appendleft(callback)


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
  """Implements InvocationContext interface for EvaluationState."""

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
      package: Optional[Package] = None,
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

  def record_rule_location(self, mnemonic):
    # Record the path of non-python callers.
    s = inspect.stack()
    callers = []
    for i in range(2, min(6, len(s))):
      c = inspect.getframeinfo(s[i][0])
      if not c.filename.endswith(".py"):
        callers.append(f"{c.filename}:{c.lineno}")
    self._rule_location = (mnemonic, callers)

  def update_current_package(
      self,
      package: Optional[Package] = None,
      package_id: Optional[PackageId] = None,
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

  def resolve_source_root(
      self, repository_id: RepositoryId
  ) -> pathlib.PurePosixPath:
    return self._state.workspace.all_repositories[
        repository_id
    ].source_directory

  def resolve_output_root(
      self, repository_id: RepositoryId
  ) -> pathlib.PurePosixPath:
    return self._state.workspace.all_repositories[
        repository_id
    ].cmake_binary_dir

  @trace_exception
  def apply_repo_mapping(
      self, target_id: TargetId, mapping_repository_id: Optional[RepositoryId]
  ) -> TargetId:
    # Resolve repository mappings
    if mapping_repository_id is None:
      assert self._caller_package_id
      mapping_repository_id = self._caller_package_id.repository_id

    mapping_repo = self._state.workspace.all_repositories.get(
        mapping_repository_id
    )
    target = mapping_repo.apply_repo_mapping(target_id)

    # Resolve --bind in the active repo.
    while target in self._state.active_repo.bindings:
      target = self._state.active_repo.bindings[target]

    if self._state.workspace.verbose and target != target_id:
      print(
          f"apply_repo_mapping({target_id.as_label()}) => {target.as_label()}"
      )

    return target

  def load_library(self, target: TargetId) -> Dict[str, Any]:
    return self._state.load_library(target)

  @trace_exception
  def get_target_info(self, target_id: TargetId) -> TargetInfo:
    return self._state.get_target_info(target_id)

  def add_rule(
      self,
      rule_id: TargetId,
      impl: RuleImpl,
      outs: Optional[List[TargetId]] = None,
      visibility: Optional[List[RelativeLabel]] = None,
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


def _exec_module(source: str, filename: str, scope: Dict[str, Any]) -> None:
  """Executes Python code with the specified `scope`.

  Polyfills support for dict union operator (PEP 584) on Python 3.8.
  """
  tree = ast.parse(source, filename)
  # Apply AST transformations to support `dict.__or__`. (PEP 584)
  tree = ast.fix_missing_locations(dict_polyfill.ASTTransformer().visit(tree))
  code = compile(tree, filename=filename, mode="exec")
  exec(code, scope)  # pylint: disable=exec-used
