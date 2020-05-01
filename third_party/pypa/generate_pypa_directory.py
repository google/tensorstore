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
"""Generates bazel workspace rules for fetching Python packages from PyPI."""

import argparse
import json
import os
import subprocess


def get_target_name(package_name):
  return package_name.lower().replace("-", "_")


def get_repo_name(package_name):
  return "pypa_%s" % (get_target_name(package_name),)


def get_full_target_name(package_name):
  return "@%s//:%s" % (get_repo_name(package_name),
                       get_target_name(package_name))


def parse_package_metadata(pip_info):
  deps = []
  metadata = {}
  for line in pip_info.splitlines():
    line = line.strip()
    if not line:
      continue
    colon = line.index(":")
    key = line[:colon]
    value = line[colon + 1:].strip()
    metadata[key] = value
  if metadata["Requires"]:
    deps = sorted(x.lower() for x in metadata["Requires"].split(", "))
  metadata["Requires"] = deps
  return metadata


def get_package_info(package_names):
  all_pip_info = subprocess.check_output(["pip", "show"] +
                                         package_names).decode()
  all_metadata = {}
  for pip_info in all_pip_info.split("\n---\n"):
    metadata = parse_package_metadata(pip_info)
    all_metadata[metadata["Name"].lower()] = metadata
  return all_metadata


def write_workspace(package_name, metadata, tools_workspace):
  dir_name = get_target_name(package_name)
  repo_name = get_repo_name(package_name)
  os.makedirs(dir_name, exist_ok=True)
  bzl_path = os.path.join(dir_name, "workspace.bzl")
  with open(bzl_path, "w") as f:
    f.write("""# Copyright 2020 The TensorStore Authors
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
""")
    f.write(
        '"""Defines a third-party bazel repo for the `%s` Python package."""\n\n'
        % (package_name,))
    f.write("""load(
    \"""" + tools_workspace + """//third_party:repo.bzl",
    "third_party_python_package",
)
""")
    for dep in metadata["Requires"]:
      f.write("""load("//third_party:pypa/""" + get_target_name(dep) +
              """/workspace.bzl", repo_""" + get_repo_name(dep) + """ = "repo")
""")
    f.write("""load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
""")
    for dep in metadata["Requires"]:
      f.write("    repo_" + get_repo_name(dep) + "()\n")

    f.write("""    maybe(
        third_party_python_package,
        name = """ + json.dumps(repo_name) + """,
        target = """ + json.dumps(dir_name) + """,
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
  print("Wrote: %s" % (bzl_path,))


def generate(args):
  all_packages = list(args.package)
  if not all_packages:
    all_packages = [
        "numpy",
        # Shell
        "ipython",
        "absl-py",
        # Test
        "pytest",
        "pytest-asyncio",
        # Build
        "wheel",
        # Docs
        "sphinx",
        "sphinx-autobuild",
        "sphinx-rtd-theme",
        "jsonpointer",
        "jsonschema",
        "pyyaml",
        # Examples
        "apache-beam",
        "gin-config",
    ]
  seen_packages = set()
  while all_packages:
    cur_packages = all_packages
    all_metadata = get_package_info(cur_packages)
    all_packages = []
    for package_name in cur_packages:
      if package_name in seen_packages:
        continue
      seen_packages.add(package_name)
      metadata = all_metadata[package_name]
      write_workspace(
          package_name=package_name,
          metadata=metadata,
          tools_workspace=args.tools_workspace,
      )
      if args.recursive:
        all_packages.extend(metadata["Requires"])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("package", nargs="*")
  ap.add_argument("-r", "--recursive", action="store_true")
  ap.add_argument("--tools-workspace", default="")
  args = ap.parse_args()
  generate(args)


if __name__ == "__main__":
  main()
