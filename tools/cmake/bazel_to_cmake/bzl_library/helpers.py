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
"""Helper methods for nested bazel_to_cmake generation."""

# pylint: disable=invalid-name,missing-function-docstring,relative-beyond-top-level,g-long-lambda

import io
import os
import sys
from typing import Any, Dict, List, Optional

from ..cmake_builder import quote_path
from ..cmake_builder import quote_string
from ..cmake_target import CMakeTarget
from ..evaluation import EvaluationState
from ..starlark.bazel_target import PackageId
from ..starlark.bazel_target import remap_target_repo
from ..starlark.invocation_context import InvocationContext
from ..starlark.invocation_context import RelativeLabel
from ..workspace import Repository

_SEP = "\n        "


def update_target_mapping(
    repo: Repository,
    root_package_id: PackageId,
    kwargs: Dict[str, Any],
) -> Dict[CMakeTarget, str]:
  """Updates kwargs[cmake_target_mapping] with resolved labels."""
  cmake_name = kwargs.get("cmake_name")
  repo_mapping: Dict[str, str] = kwargs.get("repo_mapping", {})
  target_mapping: Optional[Dict[str, str]] = kwargs.get("cmake_target_mapping")
  reverse_target_mapping: Dict[CMakeTarget, str] = {}
  canonical_target_mapping: Dict[str, str] = {}
  if target_mapping:
    for relative_label, cmake_target in target_mapping.items():
      target = remap_target_repo(
          root_package_id.parse_target(relative_label), repo_mapping)
      repo.workspace.persist_cmake_name(target, cmake_name,
                                        CMakeTarget(cmake_target))
      target_str = target.as_label()
      reverse_target_mapping.setdefault(CMakeTarget(cmake_target), target_str)
      canonical_target_mapping[target_str] = cmake_target
  kwargs["cmake_target_mapping"] = canonical_target_mapping
  kwargs["_cmake_reverse_target_mapping"] = reverse_target_mapping
  return reverse_target_mapping


def write_bazel_to_cmake_cmakelists(
    _context: InvocationContext,
    _new_cmakelists: io.StringIO,
    _patch_commands: List[str],
    name: str,
    cmake_name: str,
    bazel_to_cmake: Dict[str, Any],
    cmake_target_mapping: Optional[Dict[str, str]] = None,
    build_file: Optional[RelativeLabel] = None,
    cmake_extra_build_file: Optional[RelativeLabel] = None,
    repo_mapping: Optional[Dict[str, str]] = None,
    **kwargs):
  """Writes a nested CMakeLists.txt which invokes bazel_to_cmake.py."""
  if kwargs.get("build_file_content") is not None:
    raise ValueError("build_file_content not allowed.")
  del kwargs

  workspace = _context.access(EvaluationState).workspace
  bazel_to_cmake_args = []

  if cmake_extra_build_file is not None:
    # Labelize build file.
    build_file_path = _context.get_source_file_path(
        _context.resolve_target_or_label(cmake_extra_build_file))

    assert build_file_path is not None
    quoted_build_path = quote_path(build_file_path)
    _patch_commands.append(
        f"""${{CMAKE_COMMAND}} -E copy {quoted_build_path} extraBUILD.bazel""")
    bazel_to_cmake_args.append("--extra-build=extraBUILD.bazel")

  if build_file is not None:
    # Labelize build file.
    build_file_path = _context.get_source_file_path(
        _context.resolve_target_or_label(build_file))

    assert build_file_path is not None
    quoted_build_path = quote_path(build_file_path)
    _patch_commands.append(
        f"""${{CMAKE_COMMAND}} -E copy {quoted_build_path} BUILD.bazel""")

  bazel_to_cmake_path = os.path.abspath(sys.argv[0])
  assert workspace.save_workspace is not None
  bazel_to_cmake_args.extend([
      f"--load-workspace {quote_path(workspace.save_workspace)}",
      f"--cmake-project-name {cmake_name}",
      '--cmake-binary-dir "${CMAKE_CURRENT_BINARY_DIR}"',
      f"--bazel-repo-name {name}",
      '--build-rules-output "${CMAKE_CURRENT_BINARY_DIR}/build_rules.cmake"'
  ])
  for mapped, orig in (repo_mapping or {}).items():
    bazel_to_cmake_args.append(
        f"--repo-mapping {quote_string(mapped)} {quote_string(orig)}")
  for include_package in bazel_to_cmake.get("include", []):
    bazel_to_cmake_args.append(
        quote_string("--include-package=" + include_package))
  for exclude_package in bazel_to_cmake.get("exclude", []):
    bazel_to_cmake_args.append(
        quote_string("--exclude-package=" + exclude_package))
  if bazel_to_cmake.get("aliased_targets_only"):
    for target in (cmake_target_mapping or {}):
      bazel_to_cmake_args.append(f"--target {quote_string(target)}")
  bazel_to_cmake_args.extend(bazel_to_cmake.get("args", []))

  # NOTE: Aliases from cmake_target_mappings are inserted into the Workspace
  # when the top-level bazel / cmake package is loaded.

  _new_cmakelists.write(f"""
project({quote_string(cmake_name)})
execute_process(
  COMMAND ${{Python3_EXECUTABLE}} {quote_path(bazel_to_cmake_path)}
        {_SEP.join(bazel_to_cmake_args)}
  WORKING_DIRECTORY "${{CMAKE_CURRENT_SOURCE_DIR}}"
  COMMAND_ERROR_IS_FATAL ANY)
include("${{CMAKE_CURRENT_BINARY_DIR}}/build_rules.cmake")
""")
