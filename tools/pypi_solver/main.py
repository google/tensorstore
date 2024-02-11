# Copyright 2023 The TensorStore Authors
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

# pylint: disable=g-importing-member

"""Resolves a set of pypa requirements into specific versions.

This script applies the following steps:
1. Starting with an initial set of Requirements
  (found in *_requirements.txt files).

2. Fetch the available versions from pypa, along with the dependencies.
   2a. Fetch package metadata from the json api
   2b. Fetch package/version dependencies from the PyPa public bigquery datasets
   2c. Fetch remaining package/version data from the json API.

3. Repeatedly visit the Requirements tree adding additonal packages to resolve.

4. Once the tree is built, construct a set of constraints based on the
   package requirements using or_tools.

5. Evaluate the constraints to determine the set of packages.

This script fetches dependency information from PyPI and from the PyPa
public bigquery tables.

Note depending on how packages are created and uploaded to PyPI, PyPI may not
contain correct dependency information.  (In that case, it would be necessary to
actually install the package with `pip` in order to determine the dependencies.)
However, for all of the packages required by TensorStore, the dependency
information is available from PyPI.

# pip install google-cloud-bigquery ortools pandas requests

Beyond resolving requirements text files into packages, this tool can also
indicate which versions of each package dependency might be used to satisfy two
requirements.

$ pypi_solver/main.py -r localstack[runtime] apache-beam
"""

import argparse
import os
import pickle
from typing import Dict, List, Set

import greedy_solver
import ortools_solver
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.utils import NormalizedName
from packaging.version import Version
import pypi_downloader
import utils
import workspace_builder


def _parse_requirements_file(filename: str) -> List[str]:
  """Parses a requirements.txt file into a list of requirement strings."""
  reqs: List[str] = []
  with open(filename, "r") as f:
    for line in f.read().splitlines():
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      reqs.append(line)
  return reqs


def write_frozen_requirements(
    filename: str, solution: Dict[NormalizedName, Version]
):
  """Writes a frozen requirements file based an original requirements file."""
  assert filename.endswith(".txt")
  frozen_filename = filename[:-4] + "_frozen.txt"
  basename = os.path.basename(filename)
  with open(frozen_filename, "w") as f:
    f.write(f"# DO NOT EDIT: Generated from {basename} by pypi_solver\n")
    for req_text in _parse_requirements_file(filename):
      req = Requirement(req_text)
      if req.marker is not None:
        raise ValueError(f"Markers not supported for {req_text}")
      req.specifier = None
      req.marker = None
      v = solution.get(canonicalize_name(req.name), None)
      if v:
        f.write(f"{req}=={str(v)}\n")
      else:
        f.write(f"{req_text}\n")


def print_solution(
    parent: utils.NameAndVersion, children: List[utils.NameAndVersion]
):
  print(f"\n{parent.name}=={parent.version}")
  for c in children:
    print(f"    {c.name}=={c.version}")


def generate(args: argparse.Namespace):
  """Generates the pypa workspace.bzl file and the _frozen.txt files."""

  initial_requirements: Dict[str, Requirement] = dict()
  workspace_requirements: Set[Requirement] = set()
  override_requirements: utils.OverrideDict = dict()

  # Construct overrides from inputs.
  for line in args.requirements:
    r = Requirement(line)
    initial_requirements[r.name] = r
    if r.specifier is not None:
      print(f"Forcing {r.name} to {str(r)}")
      override_requirements[canonicalize_name(r.name)] = r

  # Add additional requirements from requirements.txt file
  def _add_initial_requirement(line: str) -> Requirement:
    nonlocal initial_requirements
    r = Requirement(line)
    if r.name not in initial_requirements:
      initial_requirements[r.name] = r
    else:
      utils.merge_requirements(initial_requirements[r.name], r)
    return r

  for filename in args.files:
    print(f"Loading requirements from: {filename}")
    is_workspace_requirement = "wheel" not in filename
    for line in _parse_requirements_file(filename):
      if is_workspace_requirement:
        workspace_requirements.add(_add_initial_requirement(line))
      else:
        _add_initial_requirement(line)

  if not initial_requirements:
    print("No requirements found.")
    return

  if args.load_metadata:
    try:
      print(f"Loading metadata pickle from {args.load_metadata}")
      with open(args.load_metadata, "rb") as f:
        metadata = pickle.load(f)
        assert isinstance(metadata, utils.PypiMetadata)
    except FileNotFoundError:
      print("Failed to load metadata pickle.")
      metadata = utils.PypiMetadata()
    except EOFError:
      print("Failed to load metadata pickle.")
      metadata = utils.PypiMetadata()
  else:
    metadata = utils.PypiMetadata()

  if not args.skip_download:
    pypi_downloader.download_metadata(
        metadata,
        initial_requirements.values(),
        args.load_metadata and args.refresh_metadata,
        args.bigquery_threshold,
        args.project,
    )
    if args.save_metadata and not args.skip_download:
      print(f"Saving metadata pickle to {args.save_metadata}")
      with open(args.save_metadata, "wb") as f:
        pickle.dump(metadata, f)

  metadata.set_override(override_requirements)

  if args.print_metadata:
    print()
    print(metadata.dump())

  solution = None

  # Start with a greedy solution.
  if not args.skip_greedy:
    try:
      solution = greedy_solver.solve_requirements(
          metadata,
          initial_requirements.values(),
      )
    except ValueError:
      solution = None

  # That didn't work, use the ortools solver.
  if not solution:
    try:
      solution = ortools_solver.solve_requirements(
          metadata,
          initial_requirements.values(),
      )
    except ValueError:
      solution = None

  # No solution. Too bad. Exit early
  if not solution:
    return 1

  # Output the current solution
  print(
      utils.print_solution(
          metadata,
          initial_requirements.values(),
          solution,
      )
  )

  # Write the frozen requirements and the workspace.
  if not args.no_freeze:
    for filename in args.files:
      write_frozen_requirements(filename, solution)

  if args.workspace:
    assert args.workspace.endswith(".bzl")
    txt = workspace_builder.build_workspace(
        metadata,
        workspace_requirements,
        solution,
        args.tools_workspace,
    )
    with open(args.workspace, "wb") as f:
      f.write(txt.encode("utf-8"))

  return 0


def main():
  ap = argparse.ArgumentParser(
      description=(
          "Resolves requirements.txt files into a bazel workspace.bzl file"
      )
  )
  ap.add_argument("files", nargs="*", default=[])
  ap.add_argument(
      "--load-metadata",
      default="/tmp/pypi_solver_metadata.pkl",
      help="Metadata pickle file to load",
  )
  ap.add_argument(
      "--save-metadata",
      default="/tmp/pypi_solver_metadata.pkl",
      help="Metadata pickle file to save",
  )
  ap.add_argument(
      "--print-metadata",
      default=False,
      action="store_true",
      help="Print metadata after loading",
  )
  ap.add_argument(
      "-r",
      "--requirements",
      action="append",
      default=[],
      type=str,
      help="Override or add additional requirements",
  )
  ap.add_argument(
      "--no-freeze",
      default=False,
      action="store_true",
      help="Avoid writing freeze files",
  )
  ap.add_argument(
      "--workspace", help="Path to the workspace.bzl file to generate"
  )
  ap.add_argument(
      "--tools-workspace", default="", help="Tools workspace bazel repository"
  )
  ap.add_argument(
      "--skip-download",
      default=False,
      action="store_true",
      help="Skip downloading additional metadata",
  )
  ap.add_argument(
      "--refresh-metadata",
      default=False,
      action="store_true",
      help="Refresh versions for all known packages",
  )
  ap.add_argument(
      "--skip-greedy",
      default=False,
      action="store_true",
      help="Skip the initial greedy solution and rely on the ortools solver",
  )
  ap.add_argument(
      "--project",
      type=str,
      help="GCS project name used for bigquery client",
  )
  ap.add_argument(
      "--bigquery-threshold",
      default=8,
      type=int,
      help="Bigquery threshold for downloading version metadata",
  )
  args = ap.parse_args()
  if not args.files and not args.requirements:
    ap.print_help()
    return 2

  return generate(args)


if __name__ == "__main__":
  main()
