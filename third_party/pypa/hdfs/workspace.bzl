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
"""Defines a third-party bazel repo for the `hdfs` Python package."""

load(
    "//third_party:repo.bzl",
    "third_party_python_package",
)
load("//third_party:pypa/docopt/workspace.bzl", repo_pypa_docopt = "repo")
load("//third_party:pypa/requests/workspace.bzl", repo_pypa_requests = "repo")
load("//third_party:pypa/six/workspace.bzl", repo_pypa_six = "repo")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    repo_pypa_docopt()
    repo_pypa_requests()
    repo_pypa_six()
    maybe(
        third_party_python_package,
        name = "pypa_hdfs",
        target = "hdfs",
        requirement = "hdfs==2.5.8",
        deps = [
            "@pypa_docopt//:docopt",
            "@pypa_requests//:requests",
            "@pypa_six//:six",
        ],
    )
