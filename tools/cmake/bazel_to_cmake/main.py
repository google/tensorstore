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
import pickle
import sys

from . import cmake_builder
from . import native_rules  # pylint: disable=unused-import
from . import rule  # pylint: disable=unused-import
from .bzl_library import default as _  # pylint: disable=unused-import
from .evaluation import EvaluationContext
from .platforms import add_platform_constraints
from .util import get_matching_build_files
from .workspace import Repository
from .workspace import Workspace


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

  # Used for sub-projects only.
  ap.add_argument("--load-workspace")
  ap.add_argument("--target", action="append", default=[])
  ap.add_argument("--target-alias", nargs=2, action="append", default=[])

  # Used for the top-level project only.
  ap.add_argument("--save-workspace")
  ap.add_argument("--define", action="append", default=[])
  ap.add_argument("--ignore-library", action="append", default=[])
  ap.add_argument("--cmake-vars")
  ap.add_argument("--bazelrc", action="append", default=[])
  ap.add_argument("--module", action="append", default=[])

  args = ap.parse_args()

  if args.load_workspace:
    # This is a dependency.  Load the workspace from the top-level project in
    # order to be able to access targets (such as `config_setting` targets) that
    # it defined, and to load `.bzl` libraries from it.  Note that `.bzl`
    # libraries themselves are not stored in the pickled data, simply the source
    # directory path of the top-level repository.
    with open(args.load_workspace, "rb") as f:
      workspace = pickle.load(f)
      assert isinstance(workspace, Workspace)
  else:
    assert args.cmake_vars is not None
    try:
      with open(args.cmake_vars, "r", encoding="utf-8") as f:
        cmake_vars = json.load(f)
    except Exception as e:
      raise ValueError(
          f"Failed to decode cmake_vars as JSON: {args.cmake_vars}") from e
    assert isinstance(cmake_vars, dict)

    workspace = Workspace(cmake_vars=cmake_vars)
    add_platform_constraints(workspace)
    workspace.values.update(("define", x) for x in args.define)

    for bazelrc in args.bazelrc:
      workspace.load_bazelrc(bazelrc)

    for module in args.module:
      workspace.add_module(module)

  repo = Repository(
      workspace=workspace,
      source_directory=os.getcwd(),
      bazel_repo_name=args.bazel_repo_name,
      cmake_project_name=args.cmake_project_name,
      cmake_binary_dir=args.cmake_binary_dir,
      top_level=args.save_workspace is not None,
  )

  if args.load_workspace:
    workspace.exclude_repo_targets(repo.bazel_repo_name)

  workspace.bazel_to_cmake_deps[repo.bazel_repo_name] = repo.cmake_project_name
  for target in args.ignore_library:
    workspace.ignore_library(repo.get_label(target))

  context = EvaluationContext(repo, save_workspace=args.save_workspace)
  context.target_aliases.update(dict(args.target_alias))
  builder = context.builder

  for x, y in args.repo_mapping:
    assert x.startswith("@")
    assert y.startswith("@")
    repo.repo_mapping[x[1:]] = y[1:]

  if repo.top_level:
    # Load the WORKSPACE
    context.process_workspace()

  # Load build files.
  include_packages = args.include_package or ["**"]
  exclude_packages = args.exclude_package or []
  build_files = get_matching_build_files(
      root_dir=repo.source_directory,
      include_packages=include_packages,
      exclude_packages=exclude_packages)
  if not build_files:
    raise ValueError(f"No build files in {repo.source_directory!r} match " +
                     f"include_packages={include_packages!r} and " +
                     f"exclude_packages={exclude_packages!r}")
  for build_file in build_files:
    context.process_build_file(build_file)

  if args.target:
    context.analyze([repo.get_label(target) for target in args.target])
  else:
    context.analyze_default_targets()

  input_files = set(context.loaded_files)

  # Add bazel_to_cmake's own source files to the list of input files.
  input_files.add(os.path.abspath(__file__))
  for module_name, module in sys.modules.items():
    if not module_name.startswith("bazel_to_cmake."):
      continue
    input_files.add(module.__file__)

  # Mark all of the Bazel files and `bazel_to_cmake` itself as dependencies of
  # the CMake configure step.  If any of those files change, the CMake configure
  # step will be re-run before building any target.
  builder.addtext(
      f"set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS {cmake_builder.quote_list(sorted(input_files))})\n",
      section=0,
  )

  if context.errors:
    print("\n".join(context.errors), file=sys.stderr)
    return 1

  if args.build_rules_output:
    with open(args.build_rules_output, "w", encoding="utf-8") as f:
      f.write(builder.as_text())

  if args.save_workspace:
    with open(args.save_workspace, "wb") as f:
      pickle.dump(workspace, f)

  return 0
