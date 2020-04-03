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
"""Defines a third-party bazel repo for the `traitlets` Python package."""

load(
    "//third_party:repo.bzl",
    "third_party_python_package",
)
load("//third_party:pypa/decorator/workspace.bzl", repo_pypa_decorator = "repo")
load("//third_party:pypa/ipython_genutils/workspace.bzl", repo_pypa_ipython_genutils = "repo")
load("//third_party:pypa/six/workspace.bzl", repo_pypa_six = "repo")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    repo_pypa_decorator()
    repo_pypa_ipython_genutils()
    repo_pypa_six()
    maybe(
        third_party_python_package,
        name = "pypa_traitlets",
        target = "traitlets",
        requirement = "traitlets==4.3.3",
        deps = [
            "@pypa_decorator//:decorator",
            "@pypa_ipython_genutils//:ipython_genutils",
            "@pypa_six//:six",
        ],
    )
