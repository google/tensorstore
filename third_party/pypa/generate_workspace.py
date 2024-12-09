#!/usr/bin/env python3

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

"""Resolves Python package requirements.

Generates "*_requirements_frozen.txt" files (used for CI) and "workspace.bzl"
used by Bazel.

The generated workspace.bzl creates one Bazel repository per package name, with
appropriate dependencies between targets across packages.

Platform-specific and python-version-specific requirements are handled as
follows:

- The target dependencies are always unconditional, but for combinations of
  platform or python version where a given dependency is not required, the
  corresponding package repository does not actually fetch the Python package,
  and instead just defines an empty py_library target.

- This does mean that the conditions are applied to the package as a whole
  rather than to individual dependency edges, but it avoids the need to attempt
  to translate Python dependency markers into Bazel select expressions.

Depends on the following command-line tools:
  uv
  uildifier
"""

import argparse
import collections
import json
import os
import re
import subprocess
from typing import NamedTuple, TextIO


_RESOLVED_REQUIREMENT_PATTERN = re.compile(
    r"""
(?P<name>[^=]+)==(?P<version>[^\s]+)
(?:\s+;\s+(?P<markers>.*?))?
(?P<hashes>(?:\s+--hash=[^ ]+)+)
\n\s+[#][ ]via[ ](?P<via>.*)\n
""",
    re.VERBOSE,
)


def resolve_requirements(
    requirements_files: list[str],
    python_version: str,
    constraints: str | None = None,
):
  """Returns the resolved requirements as text."""
  return subprocess.run(
      ["uv", "pip", "compile"]
      + requirements_files
      + [
          "--generate-hashes",
          "--universal",
          "-p",
          python_version,
          "--only-binary",
          ":all:",
          "--annotation-style",
          "line",
      ]
      + (
          ["--constraint", "/dev/stdin"]
          if constraints is not None
          else ["--no-header"]
      ),
      encoding="utf-8",
      input=constraints,
      stdout=subprocess.PIPE,
      check=True,
  ).stdout


class Requirement(NamedTuple):
  name: str
  version: str
  hashes: str
  markers: str | None
  deps: list[str]


def parse_requirements(resolved: str) -> list[Requirement]:
  """Parses the output of `resolve_requirements`."""

  # Eliminate backslash-escaped newlines to simplify parsing.
  resolved = resolved.replace("\\\n", "")

  deps = collections.defaultdict(set)
  offset = 0
  reqs = []
  while offset != len(resolved):
    m = _RESOLVED_REQUIREMENT_PATTERN.match(resolved, offset)
    if m is None:
      raise ValueError(
          "Failed to match requirement starting at: " + repr(resolved[offset:])
      )
    name = m.group("name")
    reqs.append(
        Requirement(
            name=name,
            version=m.group("version"),
            hashes=m.group("hashes"),
            markers=m.group("markers"),
            deps=[],
        )
    )
    via = m.group("via")
    if via is not None:
      for pkg in via.split(", "):
        deps[pkg].add(name)
    offset = m.end(0)

  for req in reqs:
    req.deps.extend(sorted(deps[req.name]))
  return reqs


def write_frozen_requirements(
    filename: str, python_version: str, all_reqs: str
):
  """Writes a frozen requirements file.

  Re-resolves a single requirements file, subject to `all_reqs` as constraints,
  which contains the jointly-resolved requirements. This ensures a consistent
  result.
  """
  assert filename.endswith(".txt")

  resolved = resolve_requirements(
      [filename], constraints=all_reqs, python_version=python_version
  )

  frozen_filename = filename[:-4] + "_frozen.txt"
  basename = os.path.basename(filename)

  with open(frozen_filename, "w") as f:
    f.write(
        f"# DO NOT EDIT: Generated from {basename} by generate_workspace.py\n"
    )
    f.write(resolved)


def _get_target_name(package_name):
  return re.sub("[^0-9a-z_]+", "_", package_name.lower())


def _get_repo_name(package_name):
  return "pypa_%s" % (_get_target_name(package_name),)


def _get_full_target_name(package_name) -> str:
  return "@%s//:%s" % (
      _get_repo_name(package_name),
      _get_target_name(package_name),
  )


_REPO_MACRO_BODY = """    maybe(
        third_party_python_package,
        name = %s,
        target = %s,
        requirement = %s,
"""


def _write_repo_macro(pkg: Requirement, f: TextIO):
  """Writes an individual  repo_pypa_... for a given repository."""

  package_name = _get_target_name(pkg.name)
  repo_name = _get_repo_name(pkg.name)

  f.write(f"def repo_{repo_name}():\n")

  for c in pkg.deps:
    f.write(f"    repo_{_get_repo_name(c)}()\n")

  req_terms = [f"{pkg.name}=={pkg.version}"]
  if pkg.markers is not None:
    req_terms.append(f"; {pkg.markers}")
  req_terms.extend(pkg.hashes.split())

  req_terms_formatted = (
      "[\n" + ",".join(json.dumps(term) for term in req_terms) + "]"
  )

  f.write(
      _REPO_MACRO_BODY
      % (
          json.dumps(repo_name),
          json.dumps(package_name),
          req_terms_formatted,
      )
  )

  if pkg.deps:
    f.write(
        "deps=[\n"
        + ",".join(json.dumps(_get_full_target_name(dep)) for dep in pkg.deps)
        + "],"
    )

  f.write(")\n")


_WORKSPACE_OPEN = '''# DO NOT EDIT: Generated by generate_workspace.py
"""Defines third-party bazel repos for Python packages fetched with pip."""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "%s//third_party:repo.bzl",
    "third_party_python_package",
)

def repo():
'''


def write_workspace(all_reqs: str, workspace: str, tools_workspace: str):
  """Writes the workspace.bzl file."""
  parsed_reqs = parse_requirements(all_reqs)
  parsed_reqs.sort(key=lambda req: req.name)

  with open(workspace, "w") as f:
    f.write(_WORKSPACE_OPEN % (tools_workspace,))
    for pkg in parsed_reqs:
      dep_repo_name = _get_repo_name(pkg.name)
      f.write(f"    repo_{dep_repo_name}()\n")

    for pkg in parsed_reqs:
      f.write("\n")
      _write_repo_macro(pkg, f)

  subprocess.run(["buildifier", workspace], check=True)


def generate(
    requirements_files: list[str],
    freeze: bool,
    workspace: str | None,
    python_version: str,
    tools_workspace: str,
):
  """Resolves requirements and generates output files."""

  # First resolve all requirements jointly to ensure a consistent set of
  # versions is chosen.
  all_reqs = resolve_requirements(
      requirements_files, python_version=python_version
  )

  if freeze:
    for requirements_file in requirements_files:
      write_frozen_requirements(
          requirements_file, python_version=python_version, all_reqs=all_reqs
      )

  if workspace is not None:
    write_workspace(
        all_reqs=all_reqs, workspace=workspace, tools_workspace=tools_workspace
    )


def main():
  ap = argparse.ArgumentParser(
      description=(
          "Resolves requirements.txt files into a bazel workspace.bzl file"
      )
  )
  ap.add_argument("requirements_files", nargs="*", default=[])
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
      "--python-version",
      help="Minimum python version",
      default="3.10",
  )
  ap.add_argument(
      "--tools-workspace", default="", help="Tools workspace bazel repository"
  )
  args = ap.parse_args()
  generate(
      requirements_files=args.requirements_files,
      freeze=not args.no_freeze,
      workspace=args.workspace,
      python_version=args.python_version,
      tools_workspace=args.tools_workspace,
  )


if __name__ == "__main__":
  main()
