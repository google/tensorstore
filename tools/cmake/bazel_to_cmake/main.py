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
"""Main entry point for bazel_to_cmake."""

# pylint: disable=relative-beyond-top-level

import argparse
import json
import os
import pathlib
import pickle
import sys
from typing import List, Set, Union, Dict

from . import cmake_builder
from . import native_rules  # pylint: disable=unused-import
from . import native_rules_alias  # pylint: disable=unused-import
from . import native_rules_cc  # pylint: disable=unused-import
from . import native_rules_cc_proto  # pylint: disable=unused-import
from . import native_rules_genrule  # pylint: disable=unused-import
from . import native_rules_proto  # pylint: disable=unused-import
from .bzl_library import default as _  # pylint: disable=unused-import
from .cmake_repository import CMakeRepository
from .cmake_repository import make_repo_mapping
from .cmake_target import CMakePackage
from .cmake_target import CMakeTargetPair
from .evaluation import EvaluationState
from .platforms import add_platform_constraints
from .starlark.bazel_target import RepositoryId
from .starlark.bazel_target import TargetId
from .starlark.common_providers import BuildSettingProvider
from .starlark.common_providers import ConditionProvider
from .starlark.provider import TargetInfo
from .util import get_matching_build_files
from .workspace import Repository
from .workspace import Workspace


def maybe_expand_special_targets(
    t: TargetId,
    available: Union[Set[TargetId], List[TargetId]],
):
  # Handle special targets t, :all, :... from the available targets.
  result: List[TargetId] = []
  if t.target_name == "all":
    for u in available:
      if u.package_id == t.package_id:
        result.append(u)
  elif t.target_name == "...":
    prefix = t.package_name + "/"
    for u in available:
      if u.package_name.startswith(prefix):
        result.append(u)
  else:
    result.append(t)
  return result


def get_bindings_from_args(
    repository_id: RepositoryId,
    args: argparse.Namespace,
) -> Dict[TargetId, TargetId]:
  # Add repository bindings. These provide the "native.bind" equivalent,
  # and are resolved after repo mappings. Unlike native.bind, they are
  # not restricted to only bind //external:name = alias values.
  bindings: Dict[TargetId, TargetId] = {}
  for name in args.bind:
    i = name.find("=")
    assert i > 0
    target = repository_id.get_package_id("external").parse_target(name[:i])
    actual = repository_id.parse_target(name[i + 1 :])
    if args.verbose:
      print(f"--bind {target.as_label()}={actual.as_label()}")
    assert target not in bindings
    bindings[target] = actual
  return bindings


def main():
  ap = argparse.ArgumentParser()
  # Used for top-level project and dependencies.
  ap.add_argument("--bazel-repo-name", required=True)
  ap.add_argument("--cmake-project-name", required=True)
  ap.add_argument("--build-rules-output")
  ap.add_argument("--cmake-binary-dir")
  ap.add_argument("--include-package", action="append", default=[])
  ap.add_argument("--exclude-package", action="append", default=[])
  ap.add_argument("--repo-mapping", nargs=2, action="append", default=[])
  ap.add_argument("--extra-build", action="append", default=[])
  ap.add_argument("--exclude-target", action="append", default=[])
  ap.add_argument("--bind", action="append", default=[])

  # Used for sub-projects only.
  ap.add_argument("--load-workspace")
  ap.add_argument("--target", action="append", default=[])

  # Used for the top-level project only.
  ap.add_argument("--save-workspace")
  ap.add_argument("--define", action="append", default=[])
  ap.add_argument("--ignore-library", action="append", default=[])
  ap.add_argument("--cmake-vars")
  ap.add_argument("--bazelrc", action="append", default=[])
  ap.add_argument("--module", action="append", default=[])
  ap.add_argument("--verbose", type=int, default=0)

  args = ap.parse_args()

  assert args.bazel_repo_name
  repository_id: RepositoryId = RepositoryId(args.bazel_repo_name)
  current_repository: CMakeRepository = CMakeRepository(
      repository_id=repository_id,
      cmake_project_name=CMakePackage(args.cmake_project_name),
      source_directory=pathlib.PurePath(os.getcwd()),
      cmake_binary_dir=pathlib.PurePath(args.cmake_binary_dir),
      repo_mapping=make_repo_mapping(repository_id, args.repo_mapping),
      persisted_canonical_name={},
  )

  if args.load_workspace:
    # This is a dependency.  Load the workspace from the top-level project in
    # order to be able to access targets (such as `config_setting` targets) that
    # it defined, and to load `.bzl` libraries from it.  Note that `.bzl`
    # libraries themselves are not stored in the pickled data, simply the source
    # directory path of the top-level repository.
    with open(args.load_workspace, "rb") as f:
      workspace = pickle.load(f)
      assert isinstance(workspace, Workspace)

    assert repository_id in workspace.all_repositories
    loaded: CMakeRepository = workspace.all_repositories[repository_id]

    assert loaded.cmake_project_name == current_repository.cmake_project_name
    if (
        loaded.source_directory.as_posix()
        != current_repository.source_directory.as_posix()
        or loaded.cmake_binary_dir.as_posix()
        != current_repository.cmake_binary_dir.as_posix()
    ):
      print(
          "WARNING: bazel_to_cmake repository configuration mismatch for:"
          f" {repository_id}\n"
          f"From workspace:\n{loaded}\n"
          f"From commandline:\n{current_repository}"
      )
      assert False

    current_repository = loaded
  else:
    # This is the root repository.
    assert args.cmake_vars is not None
    try:
      with open(args.cmake_vars, "r", encoding="utf-8") as f:
        cmake_vars = json.load(f)
    except Exception as e:
      raise ValueError(
          f"Failed to decode cmake_vars as JSON: {args.cmake_vars}"
      ) from e
    assert isinstance(cmake_vars, dict)

    workspace = Workspace(
        root_repository=current_repository,
        cmake_vars=cmake_vars,
        save_workspace=args.save_workspace,
    )
    add_platform_constraints(workspace)
    workspace.values.update(("define", x) for x in args.define)

    for bazelrc in args.bazelrc:
      workspace.load_bazelrc(bazelrc)

    for module in args.module:
      workspace.add_module(module)

  # pylint: disable-next=protected-access
  if args.verbose > workspace._verbose:
    # pylint: disable-next=protected-access
    workspace._verbose = args.verbose

  workspace.load_modules()

  for target in args.ignore_library:
    workspace.ignore_library(repository_id.parse_target(target))

  active_repo = Repository(
      workspace=workspace,
      repository=current_repository,
      bindings=get_bindings_from_args(repository_id, args),
      top_level=args.save_workspace is not None,
  )
  state = EvaluationState(active_repo)

  if active_repo.top_level:
    # Load the WORKSPACE file
    state.process_workspace()

  # Load build files.
  include_packages = args.include_package or ["**"]
  exclude_packages = args.exclude_package or []
  build_files = get_matching_build_files(
      root_dir=active_repo.source_directory,
      include_packages=include_packages,
      exclude_packages=exclude_packages,
  )
  if args.extra_build:
    build_files.extend(args.extra_build)

  if not build_files:
    raise ValueError(
        f"No build files in {active_repo.source_directory!r} match "
        + f"include_packages={include_packages!r} and "
        + f"exclude_packages={exclude_packages!r}"
    )
  for build_file in build_files:
    state.process_build_file(pathlib.PurePath(build_file))

  # Analyze the requested or default targets.
  default_targets_to_analyze = set(state.targets_to_analyze)
  if args.target:
    targets_to_analyze = set()
    for t in args.target:
      targets_to_analyze.update(
          maybe_expand_special_targets(
              active_repo.repository_id.parse_target(t),
              default_targets_to_analyze,
          )
      )
  else:
    targets_to_analyze = default_targets_to_analyze

  if args.exclude_target:
    for t in args.exclude_target:
      for u in maybe_expand_special_targets(
          active_repo.repository_id.parse_target(t), targets_to_analyze
      ):
        targets_to_analyze.discard(u)

  state.analyze(sorted(targets_to_analyze))

  # In verbose mode, print any global targets that have not been analyzed.
  if workspace.verbose and args.target:
    missing = []
    for t in active_repo.repository.persisted_canonical_name.keys():
      if t not in targets_to_analyze:
        missing.append(t.as_label())
    if missing:
      missing = " ".join(missing)
      print(f"--targets missing: {missing}")

  input_files = set(state.loaded_files)
  builder = state.builder

  # Add bazel_to_cmake's own source files to the list of input files.
  input_files.add(os.path.abspath(__file__))
  for module_name, module in sys.modules.items():
    if not module_name.startswith("bazel_to_cmake."):
      continue
    input_files.add(module.__file__)

  # Mark all of the Bazel files and `bazel_to_cmake` itself as dependencies of
  # the CMake configure step.  If any of those files change, the CMake configure
  # step will be re-run before building any target.
  sep = "\n    "
  builder.addtext(
      "set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "
      f"{cmake_builder.quote_list(sorted(input_files), separator=sep)})\n",
      section=0,
  )

  if state.errors:
    error_str = "\n".join(state.errors)
    print(
        f"""
---------------------------------------------------
bazel_to_cmake.py encountered errors
---------------------------------------------------
{error_str}
""",
        file=sys.stderr,
    )
    return 1

  if args.build_rules_output:
    with open(args.build_rules_output, "w", encoding="utf-8") as f:
      f.write(builder.as_text())

  if args.save_workspace:
    # Before saving the workspace, persistent targets to the workspace.
    # In order to generate consistent target names persist the following:
    # * Build and configuration settings.
    def _persist_targetinfo(target: TargetId, info: TargetInfo):
      if (
          info.get(BuildSettingProvider) is not None
          or info.get(ConditionProvider) is not None
      ):
        workspace.set_persistent_target_info(target, info)

    state.visit_analyzed_targets(_persist_targetinfo)

    # * third_party cmake target names.
    def _persist_cmakepairs(target: TargetId, cmake_pair: CMakeTargetPair):
      if target.repository_id != current_repository.repository_id:
        workspace.all_repositories[
            target.repository_id
        ].set_persisted_canonical_name(target, cmake_pair)

    state.visit_required_dep_targets(_persist_cmakepairs)

    # Validate that all repositories have source and cmake directories set
    # prior to pickling, then clear the active_repository to avoid pickling.
    for _, x in workspace.all_repositories.items():
      assert x.source_directory
      assert x.cmake_binary_dir

    workspace.active_repository = None
    with open(args.save_workspace, "wb") as f:
      pickle.dump(workspace, f)

  return 0
