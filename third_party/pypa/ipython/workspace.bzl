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
"""Defines a third-party bazel repo for the `ipython` Python package."""

load(
    "//third_party:repo.bzl",
    "third_party_python_package",
)
load("//third_party:pypa/backcall/workspace.bzl", repo_pypa_backcall = "repo")
load("//third_party:pypa/decorator/workspace.bzl", repo_pypa_decorator = "repo")
load("//third_party:pypa/jedi/workspace.bzl", repo_pypa_jedi = "repo")
load("//third_party:pypa/pexpect/workspace.bzl", repo_pypa_pexpect = "repo")
load("//third_party:pypa/pickleshare/workspace.bzl", repo_pypa_pickleshare = "repo")
load("//third_party:pypa/prompt_toolkit/workspace.bzl", repo_pypa_prompt_toolkit = "repo")
load("//third_party:pypa/pygments/workspace.bzl", repo_pypa_pygments = "repo")
load("//third_party:pypa/setuptools/workspace.bzl", repo_pypa_setuptools = "repo")
load("//third_party:pypa/traitlets/workspace.bzl", repo_pypa_traitlets = "repo")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    repo_pypa_backcall()
    repo_pypa_decorator()
    repo_pypa_jedi()
    repo_pypa_pexpect()
    repo_pypa_pickleshare()
    repo_pypa_prompt_toolkit()
    repo_pypa_pygments()
    repo_pypa_setuptools()
    repo_pypa_traitlets()
    maybe(
        third_party_python_package,
        name = "pypa_ipython",
        target = "ipython",
        requirement = "ipython==7.9.0",
        deps = [
            "@pypa_backcall//:backcall",
            "@pypa_decorator//:decorator",
            "@pypa_jedi//:jedi",
            "@pypa_pexpect//:pexpect",
            "@pypa_pickleshare//:pickleshare",
            "@pypa_prompt_toolkit//:prompt_toolkit",
            "@pypa_pygments//:pygments",
            "@pypa_setuptools//:setuptools",
            "@pypa_traitlets//:traitlets",
        ],
    )
