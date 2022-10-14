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

import importlib
import os
import pathlib
from typing import Tuple, Dict, Type, Callable, TypeVar, List, Optional, Any, Set, cast, Iterable, NamedTuple

from . import cmake_builder
from .cmake_builder import CMakeBuilder
from .configurable import Configurable
from .configurable import Select
from .configurable import SelectExpression
from .label import CMakeTarget
from .label import Label
from .label import label_to_generated_cmake_target
from .label import LabelLike
from .label import parse_label
from .label import RelativeLabel
from .label import resolve_label
from .provider import BuildSettingProvider
from .provider import CMakeDepsProvider
from .provider import CMakePackageDepsProvider
from .provider import CMakeTargetPair
from .provider import ConditionProvider
from .provider import FilesProvider
from .provider import TargetInfo
from .util import cmake_is_true
from .workspace import Repository

T = TypeVar("T")

RuleImpl = Callable[[], None]


class RuleInfo(NamedTuple):
  outs: List[Label]
  impl: RuleImpl


class EvaluationContext:
  """State used while evaluating Starlark code."""

  def __init__(self, repo: Repository, save_workspace: Optional[str] = None):
    self.repo = repo
    self.workspace = repo.workspace
    self.builder = CMakeBuilder()
    self.save_workspace = save_workspace
    self.current_package_name: Optional[str] = None
    self.current_repository_name: Optional[str] = None
    self.current_package: Optional[Package] = None
    self._processing_workspace = True
    self.loaded_files: Set[str] = set()
    self._loaded_libraries: Dict[Tuple[Label, bool], Dict[str, Any]] = dict()
    self._wrote_dummy_source = False
    self.target_aliases: Dict[Label, CMakeTarget] = {}
    self.required_dep_packages: Set[str] = set()
    self.errors: List[str] = []
    # Maps targets to their rules.
    self._unanalyzed_targets: Dict[Label, Label] = {}
    self._unanalyzed_rules: Dict[Label, RuleInfo] = {}
    self._targets_to_analyze: Set[Label] = set()
    self._call_after_workspace_loading: List[Callable[[], None]] = []
    self._call_after_analysis: List[Callable[[], None]] = []
    self.public_only = not (self.repo.top_level and cmake_is_true(
        self.workspace.cmake_vars["PROJECT_IS_TOP_LEVEL"]))

    # Ensure all rule modules are loaded
    for module in self.workspace.modules:
      importlib.import_module(module)

  def add_rule(self,
               rule_label: Label,
               impl: RuleImpl,
               outs: Optional[List[Label]] = None,
               analyze_by_default: bool = True) -> None:
    """Adds a rule.

    Args:
      rule_label: Label for the rule.
      impl: Implementation function called during analysis phase.
      outs: Additional output targets beyond `rule_label` itself.
      analyze_by_default: Whether to analyze by default, as opposed to only if
        it is a dependency of another target being analyzed.
    """
    if rule_label in self._unanalyzed_rules:
      raise ValueError(f"Duplicate rule: {rule_label}")
    self._unanalyzed_rules[rule_label] = RuleInfo(outs or [], impl)
    if outs:
      for out in outs:
        if (out in self.workspace._analyzed_targets or
            out in self._unanalyzed_targets):
          raise ValueError(f"Duplicate output: {out}")
        self._unanalyzed_targets[out] = rule_label
    self._unanalyzed_targets[rule_label] = rule_label
    if analyze_by_default:
      self._targets_to_analyze.add(rule_label)

  def add_analyzed_target(self, target: Label, info: TargetInfo) -> None:
    """Adds the `TargetInfo' for an analyzed target.

    This must be called by the `RuleImpl` function for the rule_label and each
    output target.
    """
    self.workspace._analyzed_targets[target] = info

  def get_optional_target_info(self, target: Label) -> Optional[TargetInfo]:
    analyzed_targets = self.workspace._analyzed_targets
    info = analyzed_targets.get(target)
    if info is not None:
      return info
    unanalyzed_targets = self._unanalyzed_targets
    rule_label = unanalyzed_targets.get(target)
    if rule_label is not None:
      try:
        rule_info = self._unanalyzed_rules.pop(rule_label)
        rule_info.impl()
        for target in rule_info.outs:
          unanalyzed_targets.pop(target, None)
          assert target in analyzed_targets
        unanalyzed_targets.pop(rule_label)
        assert rule_label in analyzed_targets
      except Exception as e:
        raise ValueError(f"Error analyzing {rule_label}") from e
      return analyzed_targets[target]

    # Check if it exists as a source file.
    source_path = self.get_source_file_path(target)
    if source_path is None:
      return None
    if not os.path.isfile(source_path):
      return None
    info = TargetInfo(FilesProvider([source_path]))
    analyzed_targets[target] = info
    return info

  def generate_cmake_target_pair(self,
                                 target: Label,
                                 generate_alias: bool = False
                                ) -> CMakeTargetPair:
    parsed = parse_label(target)
    repo = self.workspace.repos.get(parsed.repo_name)
    if repo is None:
      raise ValueError(f"Unknown repo in target {target}")
    cmake_project = repo.cmake_project_name
    cmake_target = label_to_generated_cmake_target(target, cmake_project, False)
    cmake_alias_target = None
    if generate_alias:
      cmake_alias_target = self.target_aliases.get(target)
      if cmake_alias_target is None:
        cmake_alias_target = label_to_generated_cmake_target(
            target, cmake_project, True)
    return CMakeTargetPair(cmake_target, cmake_alias_target)

  def get_target_info(self, target: Label) -> TargetInfo:
    info = self.get_optional_target_info(target)
    if info is None:
      raise ValueError(f"Target not found: {target}")
    return info

  def get_source_file_path(self, target: Label) -> Optional[str]:
    parsed = parse_label(target)
    repo = self.workspace.repos.get(parsed.repo_name)
    if repo is None:
      return None
    return str(
        pathlib.PurePosixPath(repo.source_directory).joinpath(
            parsed.package_name, parsed.target_name))

  def get_generated_file_path(self, target: Label) -> str:
    parsed = parse_label(target)
    repo = self.workspace.repos.get(parsed.repo_name)
    if repo is None:
      raise ValueError(f"Unknown repo name in target: {target}")
    return str(
        pathlib.PurePosixPath(repo.cmake_binary_dir).joinpath(
            parsed.package_name, parsed.target_name))

  def evaluate_condition(self, target: Label) -> bool:
    return self.get_target_info(target)[ConditionProvider].value

  def evaluate_build_setting(self, target: Label) -> Any:
    return self.get_target_info(target)[BuildSettingProvider].value

  def evaluate_configurable(self, configurable: Configurable[T]) -> T:
    """Evaluates a `Configurable` expression."""
    if isinstance(configurable, Select):
      return self._evaluate_select(configurable)
    if isinstance(configurable, SelectExpression):
      return self._evaluate_select_expression(configurable)
    return configurable

  def _evaluate_select(self, select: Select[T]) -> T:
    DEFAULT_CONDITION = "//conditions:default"
    has_default = False
    default_value = None
    matches = []
    for condition, value in select.conditions.items():
      if condition.endswith(DEFAULT_CONDITION):
        has_default = True
        default_value = value
        continue
      if self.evaluate_condition(condition):
        matches.append((condition, value))
    if len(matches) > 1:
      raise ValueError(f"More than one matching condition: {matches!r}")
    if len(matches) == 1:
      return matches[0][1]
    if has_default:
      return cast(T, default_value)
    raise ValueError("No matching condition")

  def _evaluate_select_expression(self, expr: SelectExpression[T]) -> T:
    return expr.op(
        *(self.evaluate_configurable(operand) for operand in expr.operands))

  def get_file_paths(
      self,
      target: Label,
      custom_target_deps: Optional[List[CMakeTarget]] = None,
  ) -> List[str]:
    info = self.get_target_info(target)
    if custom_target_deps is not None:
      cmake_info = info.get(CMakeDepsProvider)
      if cmake_info is not None:
        custom_target_deps.extend(cmake_info.targets)
    return info[FilesProvider].paths

  def get_targets_file_paths(
      self,
      targets: Iterable[Label],
      custom_target_deps: Optional[List[CMakeTarget]] = None,
  ) -> List[str]:
    files = []
    for target in targets:
      files.extend(self.get_file_paths(target, custom_target_deps))
    return sorted(set(files))

  def add_required_dep_package(self, package: str) -> None:
    self.required_dep_packages.add(package)

  def get_dep(self, target: Label) -> List[CMakeTarget]:
    """Maps a Bazel target to the corresponding CMake target."""
    info = self.get_optional_target_info(target)
    if info is not None:
      package_deps = info.get(CMakePackageDepsProvider)
      if package_deps is not None:
        for package in package_deps.packages:
          if package not in self.workspace.repo_cmake_packages:
            self.add_required_dep_package(package)
      return info[CMakeDepsProvider].targets

    parsed = parse_label(target)
    cmake_project_name = self.workspace.bazel_to_cmake_deps.get(
        parsed.repo_name)
    if cmake_project_name is not None:
      if cmake_project_name not in self.workspace.repo_cmake_packages:
        self.add_required_dep_package(cmake_project_name)
      return [
          label_to_generated_cmake_target(
              target, cmake_project=cmake_project_name, alias=True)
      ]

    # not handled.
    self.errors.append(f"Missing mapping for {target}")
    return []

  def get_deps(self, targets: List[Label]) -> List[CMakeTarget]:
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

  def analyze_default_targets(self):
    """Analayze targets that have been defined.

    Analyzes the transitive dependencies of all targets added with
    `always_analyze_target=True` (the default).  Currently just `filegroup`
    targets are skipped by default.
    """
    self.analyze(sorted(self._targets_to_analyze))

  def analyze(self, targets: List[Label]):
    """Analayze the transitive dependencies of `targets`.

    Note that any targets that are skipped will not be available for use as
    dependencies of targets defined in other repositories.
    """
    for target in targets:
      self.get_target_info(target)

    for package in sorted(self.required_dep_packages):
      self.builder.find_package(
          package, section=cmake_builder.FIND_DEP_PACKAGE_SECTION)

    for callback in self._call_after_analysis:
      callback()

  def get_library(self, library_target: Label) -> Dict[str, Any]:
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
    is_workspace = self._processing_workspace
    key = (library_target, is_workspace)
    library = self._loaded_libraries.get(key)
    if library is not None:
      return library
    if library_target in self.workspace.ignored_libraries:
      print(f"Ignoring library: {library_target}")
      library = self._loaded_libraries[key] = IgnoredLibrary()
      return library
    library_type = _BZL_LIBRARIES.get(key)
    if library_type is not None:
      library = self._loaded_libraries[key] = library_type(self)
      return library

    parsed = parse_label(library_target)
    if parsed.repo_name not in self.workspace.repos:
      # Unknown repository, ignore.
      library = IgnoredLibrary()
      print(f"Ignoring library: {library_target}")
      self._loaded_libraries[key] = library
      return library

    library_path = self.get_source_file_path(library_target)
    assert library_path is not None
    scope_type = BazelWorkspaceGlobals if is_workspace else BuildFileLibraryGlobals
    library = scope_type(context=self, path=library_path)
    package_name = parsed.package_name
    old_package_name = self.current_package_name
    old_repository_name = self.current_repository_name
    old_package = self.current_package
    self.current_package = None
    self.current_package_name = package_name
    self.current_repository_name = parsed.repo_name
    self._exec(library_path, library)
    self.current_package = old_package
    self.current_package_name = old_package_name
    self.current_repository_name = old_repository_name
    self._loaded_libraries[key] = library
    return library

  def _exec(self, path: str, scope: Dict[str, Any]):
    print(f"Loading {path}")
    self.loaded_files.add(path)
    code = compile(pathlib.Path(path).read_text(encoding="utf-8"), path, "exec")
    exec(code, scope)  # pylint: disable=exec-used

  def process_build_file(self, build_file: str):
    """Processes a single package (BUILD file)."""
    self._processing_workspace = False
    build_file_path = str(
        pathlib.PurePosixPath(self.repo.source_directory).joinpath(build_file))
    package_name = "/".join(build_file.split("/")[:-1])  # remove BUILD.
    package = Package(self, self.repo, package_name)
    self.current_package_name = package_name
    self.current_repository_name = self.repo.bazel_repo_name
    self.current_package = package

    scope = BuildFileGlobals(context=self, path=build_file_path)
    self._exec(build_file_path, scope)

  def process_workspace(self):
    """Processes the WORKSPACE."""
    assert self.repo.top_level
    self._processing_workspace = True
    workspace_file_path = str(
        pathlib.PurePosixPath(
            self.repo.source_directory).joinpath("WORKSPACE.bazel"))
    if not os.path.exists(workspace_file_path):
      workspace_file_path = str(
          pathlib.PurePosixPath(
              self.repo.source_directory).joinpath("WORKSPACE"))
    self.current_package_name = ""
    self.current_package = None
    scope = BazelWorkspaceGlobals(context=self, path=workspace_file_path)
    self._exec(workspace_file_path, scope)
    for callback in self._call_after_workspace_loading:
      callback()

  def call_after_workspace_loading(self, callback: Callable[[], None]) -> None:
    assert self._processing_workspace
    self._call_after_workspace_loading.append(callback)

  def call_after_analysis(self, callback: Callable[[], None]) -> None:
    self._call_after_analysis.append(callback)


class StarlarkLabel(LabelLike):
  """Corresponds to the Starlark `Label` type.

  This holds a reference to the `EvaluationContext` in order to compute
  `workspace_root`.
  """

  def __init__(self, context: EvaluationContext, target: Label):
    parsed = parse_label(target)
    self._context = context
    self.package = parsed.package_name
    self.name = parsed.target_name
    self.workspace_name = parsed.repo_name

  def __str__(self):
    return f"@{self.workspace_name}//{self.package}:{self.name}"

  def __repr__(self):
    return f"Label(\"{self}\")"

  @property
  def workspace_root(self):
    repo = self._context.workspace.repos.get(self.workspace_name)
    if repo is None:
      raise ValueError(f"Unknown bazel repo: {self.workspace_name}")
    return repo.source_directory

  def relative(self, other: str) -> "StarlarkLabel":
    repo = self._context.workspace.repos.get(self.workspace_name)
    repo_mapping = None
    if repo is not None:
      repo_mapping = repo.repo_mapping
    return StarlarkLabel(
        self._context,
        resolve_label(
            other,
            repo_mapping=repo_mapping,
            base_package=f"@{self.workspace_name}//{self.package}"))


class BazelGlobals(dict):
  """Base class for scope dict objects used when evaluating Starlark.

  Derived classes can define a `bazel_<name>` property/method to implement the
  `<name>` Starlark global.
  """

  def __init__(self, context: EvaluationContext, path: str = ""):
    self._context = context
    # For BUILD files, this is the package of the BUILD file.
    #
    # For bzl files, this is the package of the bzl file itself, not the
    # package of the BUILD file that is calling a function defined by the .bzl
    # file.  The latter is found via `self._context.current_package_name`.
    self._package_name = context.current_package_name
    self._path = path
    self._repo = context.repo
    self._workspace = context.workspace

  def __missing__(self, key):
    func = getattr(self, f"bazel_{key}")
    if func is not None:
      return func
    raise KeyError

  @property
  def repo_and_package_name(self):
    return f"@{self._repo.bazel_repo_name}//{self._package_name}"

  def bazel_Label(self, target: RelativeLabel) -> StarlarkLabel:  # pylint: disable=invalid-name
    return StarlarkLabel(
        self._context,
        resolve_label(
            target,
            repo_mapping=self._repo.repo_mapping,
            base_package=self.repo_and_package_name))

  def bazel_load(self, _library: str, *args, **kwargs):
    library_target = resolve_label(
        _library,
        repo_mapping=self._repo.repo_mapping,
        base_package=self.repo_and_package_name)

    library = self._context.get_library(library_target)
    for arg in args:
      self[arg] = library[arg]

    for key, value in kwargs.items():
      self[key] = library[value]

  bazel_hasattr = staticmethod(hasattr)
  bazel_getattr = staticmethod(getattr)


class BazelNativeWorkspaceRules:
  """Defines the `native` global accessible when evaluating workspace files."""

  def __init__(self, context: EvaluationContext):
    self._context = context

  def bind(self, *args, **kwargs):
    pass


class BazelWorkspaceGlobals(BazelGlobals):
  """Globals for WORKSPACE file and .bzl libraries loaded from the WORKSPACE."""

  def bazel_workspace(self, *args, **kwargs):
    pass

  def bazel_register_toolchains(self, *args, **kwargs):
    pass

  @property
  def bazel_native(self):
    return BazelNativeWorkspaceRules(self._context)


class BazelNativeBuildRules:
  """Defines the `native` global accessible when evaluating build files."""

  def __init__(self, context: EvaluationContext):
    self._context = context


class CcCommonModule:

  do_not_use_tools_cpp_compiler_present = True


class BuildFileLibraryGlobals(BazelGlobals):
  """Global context used for .bzl libraries loaded from BUILD files."""

  @property
  def bazel_native(self):
    return BazelNativeBuildRules(self._context)

  def bazel_select(self, conditions: Dict[str, T]) -> Select[T]:
    return Select({
        str(self.bazel_Label(condition)): value
        for condition, value in conditions.items()
    })

  @property
  def bazel_cc_common(self):
    return CcCommonModule


class BuildFileGlobals(BuildFileLibraryGlobals):
  """Global context used for BUILD files themselves."""

  def bazel_licenses(self, *args, **kwargs):
    pass

  def bazel_package(self, **kwargs):
    default_visibility = kwargs.get("default_visibility")
    if default_visibility:
      package = self._context.current_package
      assert package is not None
      package.default_visibility = package.get_label_list(default_visibility)


class IgnoredObject:

  def __call__(self, *args, **kwargs):
    return self

  def __getattr__(self, attr):
    return self


class IgnoredLibrary(dict):
  """Special globals object used for ignored libraries.

  All attributes evaluate to a no-op function.
  """

  def __missing__(self, key):
    return IgnoredObject()


_BZL_LIBRARIES: Dict[Tuple[Label, bool], Type[BazelGlobals]] = {}


def register_bzl_library(target: Label,
                         workspace: bool = False,
                         build: bool = False):

  def register(library: Type[BazelGlobals]):
    if workspace:
      _BZL_LIBRARIES[(target, True)] = library
    if build:
      _BZL_LIBRARIES[(target, False)] = library
    return library

  return register


def register_native_build_rule(impl):
  name = impl.__name__

  def wrapper(self, *args, **kwargs):
    return impl(self._context, *args, **kwargs)

  setattr(BazelNativeBuildRules, name, wrapper)
  setattr(BuildFileGlobals, f"bazel_{name}", wrapper)
  return impl


def register_native_workspace_rule(impl):
  name = impl.__name__

  def wrapper(self, *args, **kwargs):
    return impl(self._context, *args, **kwargs)

  setattr(BazelNativeWorkspaceRules, name, wrapper)
  setattr(BazelWorkspaceGlobals, f"bazel_{name}", wrapper)
  return impl


class Package:
  """Represents a Bazel package."""

  def __init__(self, context: EvaluationContext, repo: Repository,
               package_name: str):
    self.context = context
    self.repo = repo
    self.package_name = package_name
    self.package_directory = str(
        pathlib.PurePosixPath(repo.source_directory).joinpath(package_name))
    self.default_visibility: List[Label] = ["//visibility:public"]

  @property
  def workspace(self):
    return self.repo.workspace

  def get_label(self, target: RelativeLabel) -> Label:
    return resolve_label(target, self.repo.repo_mapping,
                         self.repo_and_package_name)

  def get_label_list(
      self,
      targets: Optional[Configurable[List[RelativeLabel]]] = None
  ) -> List[Label]:
    if targets is None:
      return []
    return [
        self.get_label(target)
        for target in self.context.evaluate_configurable(targets)
    ]

  @property
  def repo_and_package_name(self):
    return f"@{self.repo.bazel_repo_name}//{self.package_name}"

  def analyze_by_default(self, visibility: Optional[List[RelativeLabel]]):
    return not self.context.public_only or self.is_public(visibility)

  def analyze_test_by_default(self, visibility: Optional[List[RelativeLabel]]):
    # Treat tests as private regardless of actual visibility.
    del visibility
    return not self.context.public_only

  def is_public(self, visibility: Optional[List[RelativeLabel]]):
    if visibility is not None:
      vis_labels = self.get_label_list(visibility)
    else:
      vis_labels = self.default_visibility
    for label in vis_labels:
      if label.endswith("//visibility:public"):
        return True
    return False
