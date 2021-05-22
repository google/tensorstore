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
import json
import os

import packaging.requirements
import packaging.specifiers
import packaging.version
import requests


SUPPORTED_PYTHON_VERSIONS = ("3.6", "3.7", "3.8", "3.9")

def get_package_json(name: str):
  r = requests.get(f'https://pypi.python.org/pypi/{name}/json')
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


def _evaluate_marker(marker):
  if marker is None:
    return True
  return any(
      marker.evaluate({
          "sys_platform": _AnyValue(),
          "python_version": python_version,
          "extra": None
      }) for python_version in SUPPORTED_PYTHON_VERSIONS)


def _is_suitable_release(release):
  remaining_python_versions = set(SUPPORTED_PYTHON_VERSIONS)
  for release_pkg in release:
    requires_python = release_pkg.get('requires_python')
    if requires_python is None:
      return True
    spec = packaging.specifiers.SpecifierSet(requires_python)
    for v in list(remaining_python_versions):
      if v in spec:
        remaining_python_versions.remove(v)
    if not remaining_python_versions:
      return True
  return False


def _find_suitable_version(name, j, spec):
  releases = j["releases"]
  versions = [(packaging.version.parse(v), v) for v in releases.keys()]
  versions = [v for v in versions if not v[0].is_prerelease]
  versions.sort()
  versions.reverse()
  for (_, v_str) in versions:
    if v_str not in spec: continue
    if _is_suitable_release(releases[v_str]):
      return v_str
  return None

def get_package_metadata(req_str: str):
  req = packaging.requirements.Requirement(req_str)
  name = req.name
  j = get_package_json(name)
  version_str = _find_suitable_version(name, j, req.specifier)
  if version_str is None:
    raise ValueError(f'Could not find suitable version for {name}')
  j = get_package_json(f'{name}/{version_str}')
  requires_dist = j["info"].get("requires_dist", [])
  if requires_dist is None:
    requires_dist = []
  deps = []
  for dep_req_text in requires_dist:
    dep_req = packaging.requirements.Requirement(dep_req_text)
    if _evaluate_marker(dep_req.marker):
      deps.append(dep_req.name.lower())
  return {"Requires": sorted(deps), "Name": name, "Version": version_str}


def get_target_name(package_name):
  return package_name.lower().replace("-", "_")


def get_repo_name(package_name):
  return "pypa_%s" % (get_target_name(package_name),)


def get_full_target_name(package_name):
  return "@%s//:%s" % (get_repo_name(package_name),
                       get_target_name(package_name))


def get_package_info(package_names):
  all_metadata = {}
  with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    for metadata in executor.map(get_package_metadata, package_names):
      all_metadata[metadata["Name"].lower()] = metadata
  return all_metadata


def write_repo_macros(f, metadata):
  package_name = metadata["Name"].lower()
  repo_name = get_repo_name(package_name)
  f.write(f"""def repo_{repo_name}():
""")
  for dep in metadata["Requires"]:
    f.write(f"    repo_{get_repo_name(dep)}()\n")
  f.write("""    maybe(
        third_party_python_package,
        name = """ + json.dumps(repo_name) + """,
        target = """ + json.dumps(get_target_name(package_name)) + """,
        requirement = """ +
          json.dumps(package_name + "==" + metadata["Version"]) + """,
""")
  if metadata["Requires"]:
    f.write("        deps = [\n")
    for dep in metadata["Requires"]:
      f.write("            " + json.dumps(get_full_target_name(dep)) + ",\n")
    f.write("        ],\n")

  f.write("""    )
""")


def write_workspace(all_metadata, tools_workspace):
  all_metadata = sorted(all_metadata, key=lambda x: x["Name"].lower())
  with open("workspace.bzl", "w") as f:
    f.write(
        '"""Defines third-party bazel repos for Python packages fetched with pip."""\n\n'
    )
    f.write("""load(
    \"""" + tools_workspace + """//third_party:repo.bzl",
    "third_party_python_package",
)
""")
    f.write("""load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

""")
    f.write("""def repo():
""")
    for metadata in all_metadata:
      dep_repo_name = get_repo_name(metadata["Name"])
      f.write(f'    repo_{dep_repo_name}()\n')

    for metadata in all_metadata:
      f.write("\n")
      write_repo_macros(f, metadata)


def generate(args):
  all_packages = list(args.package)
  if not all_packages:
    all_packages = set([
        "numpy",
        # Shell
        "ipython",
        "absl-py",
        # Test
        "pytest",
        "pytest-asyncio",
        "yapf",
        # Build
        "wheel",
        # Docs
        "sphinx<4",
        "jsonschema",
        "pyyaml",
        "docutils<0.17",
        "jinja2",
        # Examples
        "apache-beam",
        "gin-config",
    ])
  seen_packages = set()
  all_metadata = {}
  while all_packages:
    cur_packages = all_packages
    all_metadata.update(get_package_info(cur_packages))
    all_packages = set()
    for req_text in cur_packages:
      req = packaging.requirements.Requirement(req_text)
      if req.name in seen_packages:
        continue
      seen_packages.add(req.name)
      metadata = all_metadata[req.name]
      all_packages.update(x for x in metadata["Requires"])
    all_packages = all_packages - seen_packages
  write_workspace(
      all_metadata=all_metadata.values(),
      tools_workspace=args.tools_workspace,
  )


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("package", nargs="*")
  ap.add_argument("--tools-workspace", default="")
  args = ap.parse_args()
  generate(args)


if __name__ == "__main__":
  main()
