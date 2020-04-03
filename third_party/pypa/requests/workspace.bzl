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
"""Defines a third-party bazel repo for the `requests` Python package."""

load(
    "//third_party:repo.bzl",
    "third_party_python_package",
)
load("//third_party:pypa/certifi/workspace.bzl", repo_pypa_certifi = "repo")
load("//third_party:pypa/chardet/workspace.bzl", repo_pypa_chardet = "repo")
load("//third_party:pypa/idna/workspace.bzl", repo_pypa_idna = "repo")
load("//third_party:pypa/urllib3/workspace.bzl", repo_pypa_urllib3 = "repo")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    repo_pypa_certifi()
    repo_pypa_chardet()
    repo_pypa_idna()
    repo_pypa_urllib3()
    maybe(
        third_party_python_package,
        name = "pypa_requests",
        target = "requests",
        requirement = "requests==2.22.0",
        deps = [
            "@pypa_certifi//:certifi",
            "@pypa_chardet//:chardet",
            "@pypa_idna//:idna",
            "@pypa_urllib3//:urllib3",
        ],
    )
