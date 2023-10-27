#!/usr/bin/env python3
# Copyright 2020 The TensorStore Authors
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
"""Generates bazel workspace rules for fetching Python packages from PyPI.

This script fetches dependency information from PyPI.

Note depending on how packages are created and uploaded to PyPI, PyPI may not
contain correct dependency information.  (In that case, it would be necessary to
actually install the package with `pip` in order to determine the dependencies.)
However, for all of the packages required by TensorStore, the dependency
information is available from PyPI.
"""

import argparse
import concurrent.futures
from dataclasses import dataclass
import functools
import json
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Sequence, Tuple

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.utils import NormalizedName
import packaging.version
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

Json = Any
PackageVersion = Tuple[NormalizedName, str]

SUPPORTED_PYTHON_VERSIONS = ("3.9", "3.10", "3.11")

BAZEL_REQUIREMENTS_FILENAMES = [
    "build_requirements.txt",
    "test_requirements.txt",
    "docs_requirements.txt",
    "doctest_requirements.txt",
    "shell_requirements.txt",
    "examples_requirements.txt",
]

CI_REQUIREMENTS_FILENAMES = [
    "cibuildwheel_requirements.txt",
    "wheel_requirements.txt",
]

# Dependencies to ignore (to avoid cycles)
#
# https://github.com/sphinx-doc/sphinx/issues/11567
IGNORED_DEPENDENCIES = {
    "sphinxcontrib-applehelp": {"sphinx"},
    "sphinxcontrib-htmlhelp": {"sphinx"},
    "sphinxcontrib-qthelp": {"sphinx"},
    "sphinxcontrib-devhelp": {"sphinx"},
    "sphinxcontrib-serializinghtml": {"sphinx"},
}


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


@functools.cache
def _get_session():
  s = requests.Session()
  retry = Retry(connect=10, read=10, backoff_factor=0.2)
  adapter = HTTPAdapter(max_retries=retry)
  s.mount("http://", adapter)
  s.mount("https://", adapter)
  return s


# https://warehouse.pypa.io/api-reference/json.html
def get_pypa_json(package_selector: str) -> Json:
  """Fetches a Python package or package/version from PyPI."""
  uri = f"https://pypi.org/pypi/{package_selector}/json"
  print(uri)
  r = _get_session().get(uri, timeout=5)
  r.raise_for_status()
  return r.json()


class _AnyValue:
  """Special value for which all comparisons return true.

  This is used for marker evaluation to accept any value.
  """

  def __eq__(self, other):
    return True

  def __ne__(self, other):
    return True

  def __lt__(self, other):
    return True

  def __gt__(self, other):
    return True

  def __ge__(self, other):
    return True

  def __le__(self, other):
    return True


def _evaluate_marker(req: Requirement, marker: Marker) -> bool:
  """Evaluates a packaging.Marker for python versions and Requirement.extras."""
  if marker is None:
    return True

  extras = req.extras if req.extras else [None]
  for v in SUPPORTED_PYTHON_VERSIONS:
    for e in extras:
      if marker.evaluate({
          "sys_platform": _AnyValue(),
          "python_version": v,
          "extra": e,
      }):
        return True
  return False


def _is_suitable_release(release: Json):
  remaining_python_versions = set(SUPPORTED_PYTHON_VERSIONS)
  for release_pkg in release:
    requires_python = release_pkg.get("requires_python")
    if requires_python is None:
      return True
    # Trim some invalid substrings
    while True:
      if requires_python.endswith(".*"):
        requires_python = requires_python[:-2]
      elif requires_python.endswith(".*"):
        requires_python = requires_python[:-1]
      else:
        break
    spec = SpecifierSet(requires_python)
    for v in list(remaining_python_versions):
      if v in spec:
        remaining_python_versions.remove(v)
    if not remaining_python_versions:
      return True
  return False


def _merge_requirements(dest: Requirement, b: Requirement) -> bool:
  """Merges Requirement object b into dest."""
  result = False
  if dest.extras or b.extras:
    extras = set()
    extras.update(dest.extras or [])
    extras.update(b.extras or [])
    extras = list(sorted(extras))
    if extras != list(sorted(dest.extras or [])):
      dest.extras = extras
      result = True
  if dest.specifier != b.specifier:
    merged = set()
    merged.update(iter(dest.specifier))
    merged.update(iter(b.specifier))
    merged_specifier = SpecifierSet(",".join([str(x) for x in merged]))
    if merged_specifier != dest.specifier:
      dest.specifier = merged_specifier
      result = True
  # TODO Merge markers?
  return result


@dataclass
class Resolved:
  package_name: str
  version: str
  deps: List[PackageVersion]


class RequirementResolver:
  """Maintains metadata required to resolve sets of Requirements.

  See: https://warehouse.pypa.io/api-reference/json.html
  """

  def __init__(self):
    self.package_versions: Dict[NormalizedName, List[str]] = {}
    self.package_version_requires: Dict[PackageVersion, List[Requirement]] = {}

  def handle_package_version_metadata(self, j: Json):
    """Process metadata for a specific PyPa package version."""
    info = j["info"]
    name = canonicalize_name(info["name"])
    version = info["version"]
    ignored_deps = IGNORED_DEPENDENCIES.get(name, [])
    reqs = []
    requires_dist = info.get("requires_dist")
    if requires_dist:
      for req_txt in requires_dist:
        r = Requirement(req_txt)
        if canonicalize_name(r.name) in ignored_deps:
          continue
        reqs.append(r)
    self.package_version_requires[(name, version)] = reqs

  def handle_package_metadata(self, j: Json):
    """Process metadata for a PyPa package."""
    info = j["info"]
    name = canonicalize_name(info["name"])
    releases = j["releases"]
    versions = []

    for v, release_list in releases.items():
      if not _is_suitable_release(release_list):
        continue
      try:
        versions.append((packaging.version.parse(v), v))
      except packaging.version.InvalidVersion:
        print(f"{name} skipping invalid version: {v}")

    # package_versions are in order from highest version to lowest, though
    # still persisted as raw strings.
    versions.sort()
    versions.reverse()
    self.package_versions[name] = [x[1] for x in versions]

    # The package metadata likely contains the information for the
    # latest build; might as well store it.
    if info.get("version") and info.get("requires_dist"):
      self.handle_package_version_metadata(j)

  def _get_package_metadata(self, package_name: NormalizedName):
    try:
      j = get_pypa_json(package_name)
      self.handle_package_metadata(j)
    except Exception as e:
      print(f"{package_name} {e}")
      traceback.print_exception(*sys.exc_info())

  def _get_version_metadata(self, package_and_version: PackageVersion):
    try:
      j = get_pypa_json(f"{package_and_version[0]}/{package_and_version[1]}")
      self.handle_package_version_metadata(j)
    except Exception as e:
      print(f"{package_and_version} {e}")
      traceback.print_exception(*sys.exc_info())

  def get_metadata(
      self,
      package_names: Sequence[NormalizedName],
      package_versions: Sequence[PackageVersion],
  ):
    # Each entry in package_name and package_version is retrieved independently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
      for x in package_names:
        if x not in self.package_versions:
          executor.submit(self._get_package_metadata, x)
      for x in package_versions:
        if x not in self.package_version_requires:
          executor.submit(self._get_version_metadata, x)

  def resolve_requirements(
      self, initial_requirements: Sequence[Requirement]
  ) -> Dict[NormalizedName, Resolved]:
    """Iteratively resolve the PyPa requirements."""

    # requirements contains all packages along with any constraints
    # which have been detected during the attempt at resolving package
    # versions.
    requirements: Dict[NormalizedName, Requirement] = {}
    for x in initial_requirements:
      requirements[canonicalize_name(x.name)] = Requirement(str(x))

    # selected contains the mapping from name to version; this is not
    # set as a constraint as it may change every loop through the package
    # resolver.
    selected: Dict[NormalizedName, str] = {}
    versions_to_resolve: List[PackageVersion] = []
    loop: bool = True

    # Repeat until quiescent.
    while loop:
      print("Retrieving metadata...")
      self.get_metadata(
          requirements.keys(),
          versions_to_resolve,
      )
      versions_to_resolve.clear()
      loop = False

      # Evaluate each known dependency.
      for r in list(requirements.values()):
        name = canonicalize_name(r.name)
        r_versions = self.package_versions.get(name)
        if r_versions is None:
          # This should never happen; all packages should have versions by now.
          raise ValueError(f"Package has no versions. {str(r)}")

        # 1. Resolve the specifier version filter to the highest version which
        # satisfies the specifier (the first item in r_versions has the highest
        # compatible version).
        suitable = list(r.specifier.filter(r_versions))
        if not suitable:
          suitable = list(r.specifier.filter(r_versions, prereleases=True))
          if not suitable:
            raise ValueError(f"Cannot satisfy package requirements: {str(r)}")

        selected[name] = suitable[0]
        version_requires = self.package_version_requires.get(
            (name, suitable[0])
        )
        if version_requires is None:
          # The first time a dependency is the version data will be unavailable.
          versions_to_resolve.append((name, suitable[0]))
          loop = True
          continue

        # 2. For the selected version, evaluate each possible dependency.
        # Dependencies are added or merged with the existing requirements.
        for r_dep in version_requires:
          if not _evaluate_marker(r, r_dep.marker):
            continue

          print(f"Package dependency {str(r)} -> {str(r_dep)}")

          r_dep_name = canonicalize_name(r_dep.name)
          if r_dep_name not in requirements:
            requirements[r_dep_name] = r_dep
            loop = True
            continue

          if _merge_requirements(requirements[r_dep_name], r_dep):
            loop = True

    # end while

    # Output the versions and dependencies for each package
    result: Dict[NormalizedName, Resolved] = {}
    for name, r in requirements.items():
      version = selected[name]
      version_requires = self.package_version_requires.get((name, version))
      deps: List[PackageVersion] = []
      for r_dep in version_requires:
        if _evaluate_marker(r, r_dep.marker):
          r_dep_name = canonicalize_name(r_dep.name)
          deps.append((r_dep_name, selected[r_dep_name]))
      result[name] = Resolved(
          package_name=name, version=selected[name], deps=deps
      )
    return result


def get_target_name(package_name):
  return re.sub("[^0-9a-z_]+", "_", package_name.lower())


def get_repo_name(package_name):
  return "pypa_%s" % (get_target_name(package_name),)


def get_full_target_name(package_name) -> str:
  return "@%s//:%s" % (
      get_repo_name(package_name),
      get_target_name(package_name),
  )


def write_repo_macros(f, metadata: Resolved):
  package_name = get_target_name(metadata.package_name)
  repo_name = get_repo_name(metadata.package_name)
  f.write(f"""def repo_{repo_name}():
""")
  for repo_dep in sorted(set(get_repo_name(dep[0]) for dep in metadata.deps)):
    f.write(f"    repo_{repo_dep}()\n")
  f.write(
      """    maybe(
        third_party_python_package,
        name = """
      + json.dumps(repo_name)
      + """,
        target = """
      + json.dumps(package_name)
      + """,
        requirement = """
      + json.dumps(metadata.package_name + "==" + metadata.version)
      + """,
"""
  )
  if metadata.deps:
    f.write("        deps = [\n")
    for unique_dep in sorted(
        set(get_full_target_name(dep[0]) for dep in metadata.deps)
    ):
      f.write("            " + json.dumps(unique_dep) + ",\n")
    f.write("        ],\n")

  f.write("""    )
""")


def write_workspace(
    resolved_metadata: Dict[str, Resolved], tools_workspace: str
):
  keys = sorted([k for k in resolved_metadata], key=lambda x: x.lower())
  with open("workspace.bzl", "w") as f:
    f.write("# DO NOT EDIT: Generated by generate_workspace.py\n")
    f.write(
        '"""Defines third-party bazel repos for Python packages fetched with'
        ' pip."""\n\n'
    )
    f.write(
        """load(
    \""""
        + tools_workspace
        + """//third_party:repo.bzl",
    "third_party_python_package",
)
"""
    )
    f.write("""load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

""")
    f.write("""def repo():
""")
    for package_name in keys:
      dep_repo_name = get_repo_name(package_name)
      f.write(f"    repo_{dep_repo_name}()\n")

    for package_name in keys:
      f.write("\n")
      write_repo_macros(f, resolved_metadata[package_name])


def write_frozen_requirements(
    filename: str, resolved_metadata: Dict[NormalizedName, Resolved]
):
  """Writes a frozen requirements file based an original requirements file."""
  frozen_filename = filename[:-4] + "_frozen.txt"
  basename = os.path.basename(filename)
  with open(frozen_filename, "w") as f:
    f.write(
        f"# DO NOT EDIT: Generated from {basename} by generate_workspace.py\n"
    )
    for req_text in _parse_requirements_file(filename):
      req = Requirement(req_text)
      if req.marker is not None:
        raise ValueError(f"Markers not supported for {req_text}")
      req.specifier = None
      req.marker = None
      x = resolved_metadata.get(canonicalize_name(req.name), None)
      if x:
        f.write(f"{req}=={x.version}\n")
      else:
        f.write(f"{req_text}\n")


def generate(args: argparse.Namespace):
  """Generates the pypa workspace.bzl file and the _frozen.txt files."""
  script_dir = os.path.dirname(__file__)

  requirements: Dict[str, Requirement] = {}
  for filename in BAZEL_REQUIREMENTS_FILENAMES:
    for line in _parse_requirements_file(os.path.join(script_dir, filename)):
      r = Requirement(line)
      if r.name not in requirements:
        requirements[r.name] = r
        continue
      _merge_requirements(requirements[r.name], r)

  resolver = RequirementResolver()
  resolved_metadata = resolver.resolve_requirements(requirements.values())
  write_workspace(
      resolved_metadata=resolved_metadata,
      tools_workspace=args.tools_workspace,
  )

  # "freeze" our selected versions.
  for r in requirements.values():
    v = resolved_metadata[canonicalize_name(r.name)].version
    r.specifier = SpecifierSet(f"=={v}")

  # Add in additional packages not required by Bazel but required for by
  # continuous integration.
  for filename in CI_REQUIREMENTS_FILENAMES:
    for line in _parse_requirements_file(os.path.join(script_dir, filename)):
      r = Requirement(line)
      if r.name not in requirements:
        requirements[r.name] = r
        continue
      _merge_requirements(requirements[r.name], r)

  resolved_metadata = resolver.resolve_requirements(requirements.values())
  for filename in BAZEL_REQUIREMENTS_FILENAMES + CI_REQUIREMENTS_FILENAMES:
    write_frozen_requirements(
        os.path.join(script_dir, filename), resolved_metadata
    )


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("package", nargs="*")
  ap.add_argument("--tools-workspace", default="")
  args = ap.parse_args()
  generate(args)


if __name__ == "__main__":
  main()
