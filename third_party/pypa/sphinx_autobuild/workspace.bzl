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
"""Defines a third-party bazel repo for the `sphinx-autobuild` Python package."""

load(
    "//third_party:repo.bzl",
    "third_party_python_package",
)
load("//third_party:pypa/argh/workspace.bzl", repo_pypa_argh = "repo")
load("//third_party:pypa/livereload/workspace.bzl", repo_pypa_livereload = "repo")
load("//third_party:pypa/pathtools/workspace.bzl", repo_pypa_pathtools = "repo")
load("//third_party:pypa/port_for/workspace.bzl", repo_pypa_port_for = "repo")
load("//third_party:pypa/pyyaml/workspace.bzl", repo_pypa_pyyaml = "repo")
load("//third_party:pypa/tornado/workspace.bzl", repo_pypa_tornado = "repo")
load("//third_party:pypa/watchdog/workspace.bzl", repo_pypa_watchdog = "repo")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    repo_pypa_argh()
    repo_pypa_livereload()
    repo_pypa_pathtools()
    repo_pypa_port_for()
    repo_pypa_pyyaml()
    repo_pypa_tornado()
    repo_pypa_watchdog()
    maybe(
        third_party_python_package,
        name = "pypa_sphinx_autobuild",
        target = "sphinx_autobuild",
        requirement = "sphinx-autobuild==0.7.1",
        deps = [
            "@pypa_argh//:argh",
            "@pypa_livereload//:livereload",
            "@pypa_pathtools//:pathtools",
            "@pypa_port_for//:port_for",
            "@pypa_pyyaml//:pyyaml",
            "@pypa_tornado//:tornado",
            "@pypa_watchdog//:watchdog",
        ],
    )
