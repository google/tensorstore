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
"""Defines a third-party bazel repo for the `jsonschema` Python package."""

load(
    "//third_party:repo.bzl",
    "third_party_python_package",
)
load("//third_party:pypa/attrs/workspace.bzl", repo_pypa_attrs = "repo")
load("//third_party:pypa/importlib_metadata/workspace.bzl", repo_pypa_importlib_metadata = "repo")
load("//third_party:pypa/pyrsistent/workspace.bzl", repo_pypa_pyrsistent = "repo")
load("//third_party:pypa/setuptools/workspace.bzl", repo_pypa_setuptools = "repo")
load("//third_party:pypa/six/workspace.bzl", repo_pypa_six = "repo")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    repo_pypa_attrs()
    repo_pypa_importlib_metadata()
    repo_pypa_pyrsistent()
    repo_pypa_setuptools()
    repo_pypa_six()
    maybe(
        third_party_python_package,
        name = "pypa_jsonschema",
        target = "jsonschema",
        requirement = "jsonschema==3.2.0",
        deps = [
            "@pypa_attrs//:attrs",
            "@pypa_importlib_metadata//:importlib_metadata",
            "@pypa_pyrsistent//:pyrsistent",
            "@pypa_setuptools//:setuptools",
            "@pypa_six//:six",
        ],
    )
