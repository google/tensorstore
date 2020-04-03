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
"""Defines a third-party bazel repo for the `pytest` Python package."""

load(
    "//third_party:repo.bzl",
    "third_party_python_package",
)
load("//third_party:pypa/atomicwrites/workspace.bzl", repo_pypa_atomicwrites = "repo")
load("//third_party:pypa/attrs/workspace.bzl", repo_pypa_attrs = "repo")
load("//third_party:pypa/importlib_metadata/workspace.bzl", repo_pypa_importlib_metadata = "repo")
load("//third_party:pypa/more_itertools/workspace.bzl", repo_pypa_more_itertools = "repo")
load("//third_party:pypa/packaging/workspace.bzl", repo_pypa_packaging = "repo")
load("//third_party:pypa/pluggy/workspace.bzl", repo_pypa_pluggy = "repo")
load("//third_party:pypa/py/workspace.bzl", repo_pypa_py = "repo")
load("//third_party:pypa/wcwidth/workspace.bzl", repo_pypa_wcwidth = "repo")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    repo_pypa_atomicwrites()
    repo_pypa_attrs()
    repo_pypa_importlib_metadata()
    repo_pypa_more_itertools()
    repo_pypa_packaging()
    repo_pypa_pluggy()
    repo_pypa_py()
    repo_pypa_wcwidth()
    maybe(
        third_party_python_package,
        name = "pypa_pytest",
        target = "pytest",
        requirement = "pytest==5.2.2",
        deps = [
            "@pypa_atomicwrites//:atomicwrites",
            "@pypa_attrs//:attrs",
            "@pypa_importlib_metadata//:importlib_metadata",
            "@pypa_more_itertools//:more_itertools",
            "@pypa_packaging//:packaging",
            "@pypa_pluggy//:pluggy",
            "@pypa_py//:py",
            "@pypa_wcwidth//:wcwidth",
        ],
    )
