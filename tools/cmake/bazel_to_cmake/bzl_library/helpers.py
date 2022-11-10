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
from typing import Dict, List, Optional, Any

from .. import cmake_builder
from ..evaluation import EvaluationContext
from ..label import CMakeTarget
from ..label import Label
from ..label import resolve_label
from ..workspace import Repository


def get_file_path(_context: EvaluationContext, _repo: Repository,
                  label: Label) -> str:
  """Returns the source filepath for a label."""
  path = _context.get_source_file_path(_repo.get_label(label))
  assert path is not None
  return path


def update_target_mapping(repo: Repository,
                          reverse_target_mapping: Dict[CMakeTarget, Label],
                          kwargs: Dict[str, Any]):
  """Updates kwargs[cmake_target_mapping] with resolved labels."""
  bazel_name = kwargs["name"]
  new_base_package = f"@{bazel_name}//"
  target_mapping = kwargs.get("cmake_target_mapping")
  canonical_target_mapping: Dict[Label, CMakeTarget] = {}
  if target_mapping:
    cmake_name = kwargs.get("cmake_name")
    for relative_label, cmake_target in target_mapping.items():
      label = resolve_label(relative_label, base_package=new_base_package)
      reverse_target_mapping.setdefault(cmake_target, label)
      repo.workspace.set_bazel_target_mapping(
          label, cmake_target, cmake_package=cmake_name)
      canonical_target_mapping[label] = cmake_target
  kwargs["cmake_target_mapping"] = canonical_target_mapping
  kwargs["_cmake_reverse_target_mapping"] = reverse_target_mapping
  return new_base_package


def write_bazel_to_cmake_cmakelists(
    _new_cmakelists: io.StringIO,
    _patch_commands: List[str],
    _context: EvaluationContext,
    _repo: Repository,
    name: str,
    cmake_name: str,
    bazel_to_cmake: Dict[str, Any],
    cmake_target_mapping: Optional[Dict[Label, CMakeTarget]] = None,
    build_file: Optional[Label] = None,
    repo_mapping: Optional[Dict[str, str]] = None,
    **kwargs):
  """Writes a nested CMakeLists.txt which invokes bazel_to_cmake.py."""
  del kwargs
  if build_file is not None:
    cmake_command = _context.workspace.cmake_vars["CMAKE_COMMAND"]
    quoted_build_path = cmake_builder.quote_path(
        get_file_path(_context, _repo, build_file))
    _patch_commands.append(
        f"""{cmake_builder.quote_path(cmake_command)} -E copy {quoted_build_path} BUILD.bazel"""
    )
  bazel_to_cmake_path = os.path.abspath(sys.argv[0])
  bazel_to_cmake_cmd = f"${{Python3_EXECUTABLE}} {cmake_builder.quote_path(bazel_to_cmake_path)}"
  assert _context.save_workspace is not None
  bazel_to_cmake_cmd += (
      f" --load-workspace {cmake_builder.quote_path(_context.save_workspace)}")
  bazel_to_cmake_cmd += f" --cmake-project-name {cmake_name}"
  bazel_to_cmake_cmd += ' --cmake-binary-dir "${CMAKE_CURRENT_BINARY_DIR}"'
  bazel_to_cmake_cmd += f" --bazel-repo-name {name}"
  bazel_to_cmake_cmd += (
      ' --build-rules-output "${CMAKE_CURRENT_BINARY_DIR}/build_rules.cmake"')
  for mapped, orig in (repo_mapping or {}).items():
    bazel_to_cmake_cmd += f" --repo-mapping {mapped} {orig}"
  for include_package in bazel_to_cmake.get("include", []):
    bazel_to_cmake_cmd += " " + cmake_builder.quote_string(
        "--include-package=" + include_package)
  for exclude_package in bazel_to_cmake.get("exclude", []):
    bazel_to_cmake_cmd += " " + cmake_builder.quote_string(
        "--exclude-package=" + exclude_package)
  if bazel_to_cmake.get("aliased_targets_only"):
    for target in (cmake_target_mapping or {}):
      bazel_to_cmake_cmd += f" --target {cmake_builder.quote_string(target)}"
  for bazel_target, cmake_alias in (cmake_target_mapping or {}).items():
    bazel_to_cmake_cmd += f" --target-alias {cmake_builder.quote_string(bazel_target)} {cmake_builder.quote_string(cmake_alias)}"

  bazel_to_cmake_cmd += " " + " ".join(bazel_to_cmake.get("args", []))
  _new_cmakelists.write(f"""
project({cmake_builder.quote_string(cmake_name)})
execute_process(
  COMMAND {bazel_to_cmake_cmd}
  WORKING_DIRECTORY "${{CMAKE_CURRENT_SOURCE_DIR}}"
  COMMAND_ERROR_IS_FATAL ANY)
include("${{CMAKE_CURRENT_BINARY_DIR}}/build_rules.cmake")
""")
