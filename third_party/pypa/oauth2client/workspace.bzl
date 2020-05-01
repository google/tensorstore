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
"""Defines a third-party bazel repo for the `oauth2client` Python package."""

load(
    "//third_party:repo.bzl",
    "third_party_python_package",
)
load("//third_party:pypa/httplib2/workspace.bzl", repo_pypa_httplib2 = "repo")
load("//third_party:pypa/pyasn1/workspace.bzl", repo_pypa_pyasn1 = "repo")
load("//third_party:pypa/pyasn1_modules/workspace.bzl", repo_pypa_pyasn1_modules = "repo")
load("//third_party:pypa/rsa/workspace.bzl", repo_pypa_rsa = "repo")
load("//third_party:pypa/six/workspace.bzl", repo_pypa_six = "repo")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    repo_pypa_httplib2()
    repo_pypa_pyasn1()
    repo_pypa_pyasn1_modules()
    repo_pypa_rsa()
    repo_pypa_six()
    maybe(
        third_party_python_package,
        name = "pypa_oauth2client",
        target = "oauth2client",
        requirement = "oauth2client==3.0.0",
        deps = [
            "@pypa_httplib2//:httplib2",
            "@pypa_pyasn1//:pyasn1",
            "@pypa_pyasn1_modules//:pyasn1_modules",
            "@pypa_rsa//:rsa",
            "@pypa_six//:six",
        ],
    )
