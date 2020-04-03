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
"""Defines a third-party bazel repo for the `sphinx` Python package."""

load(
    "//third_party:repo.bzl",
    "third_party_python_package",
)
load("//third_party:pypa/alabaster/workspace.bzl", repo_pypa_alabaster = "repo")
load("//third_party:pypa/babel/workspace.bzl", repo_pypa_babel = "repo")
load("//third_party:pypa/docutils/workspace.bzl", repo_pypa_docutils = "repo")
load("//third_party:pypa/imagesize/workspace.bzl", repo_pypa_imagesize = "repo")
load("//third_party:pypa/jinja2/workspace.bzl", repo_pypa_jinja2 = "repo")
load("//third_party:pypa/packaging/workspace.bzl", repo_pypa_packaging = "repo")
load("//third_party:pypa/pygments/workspace.bzl", repo_pypa_pygments = "repo")
load("//third_party:pypa/requests/workspace.bzl", repo_pypa_requests = "repo")
load("//third_party:pypa/setuptools/workspace.bzl", repo_pypa_setuptools = "repo")
load("//third_party:pypa/snowballstemmer/workspace.bzl", repo_pypa_snowballstemmer = "repo")
load("//third_party:pypa/sphinxcontrib_applehelp/workspace.bzl", repo_pypa_sphinxcontrib_applehelp = "repo")
load("//third_party:pypa/sphinxcontrib_devhelp/workspace.bzl", repo_pypa_sphinxcontrib_devhelp = "repo")
load("//third_party:pypa/sphinxcontrib_htmlhelp/workspace.bzl", repo_pypa_sphinxcontrib_htmlhelp = "repo")
load("//third_party:pypa/sphinxcontrib_jsmath/workspace.bzl", repo_pypa_sphinxcontrib_jsmath = "repo")
load("//third_party:pypa/sphinxcontrib_qthelp/workspace.bzl", repo_pypa_sphinxcontrib_qthelp = "repo")
load("//third_party:pypa/sphinxcontrib_serializinghtml/workspace.bzl", repo_pypa_sphinxcontrib_serializinghtml = "repo")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    repo_pypa_alabaster()
    repo_pypa_babel()
    repo_pypa_docutils()
    repo_pypa_imagesize()
    repo_pypa_jinja2()
    repo_pypa_packaging()
    repo_pypa_pygments()
    repo_pypa_requests()
    repo_pypa_setuptools()
    repo_pypa_snowballstemmer()
    repo_pypa_sphinxcontrib_applehelp()
    repo_pypa_sphinxcontrib_devhelp()
    repo_pypa_sphinxcontrib_htmlhelp()
    repo_pypa_sphinxcontrib_jsmath()
    repo_pypa_sphinxcontrib_qthelp()
    repo_pypa_sphinxcontrib_serializinghtml()
    maybe(
        third_party_python_package,
        name = "pypa_sphinx",
        target = "sphinx",
        requirement = "sphinx==2.4.1",
        deps = [
            "@pypa_alabaster//:alabaster",
            "@pypa_babel//:babel",
            "@pypa_docutils//:docutils",
            "@pypa_imagesize//:imagesize",
            "@pypa_jinja2//:jinja2",
            "@pypa_packaging//:packaging",
            "@pypa_pygments//:pygments",
            "@pypa_requests//:requests",
            "@pypa_setuptools//:setuptools",
            "@pypa_snowballstemmer//:snowballstemmer",
            "@pypa_sphinxcontrib_applehelp//:sphinxcontrib_applehelp",
            "@pypa_sphinxcontrib_devhelp//:sphinxcontrib_devhelp",
            "@pypa_sphinxcontrib_htmlhelp//:sphinxcontrib_htmlhelp",
            "@pypa_sphinxcontrib_jsmath//:sphinxcontrib_jsmath",
            "@pypa_sphinxcontrib_qthelp//:sphinxcontrib_qthelp",
            "@pypa_sphinxcontrib_serializinghtml//:sphinxcontrib_serializinghtml",
        ],
    )
