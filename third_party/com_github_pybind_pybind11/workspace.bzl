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

# buildifier: disable=module-docstring

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//third_party:repo.bzl", "third_party_http_archive")

# REPO_BRANCH = master

def repo():
    maybe(
        third_party_http_archive,
        name = "com_github_pybind_pybind11",
        strip_prefix = "pybind11-2.13.6",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/pybind/pybind11/archive/v2.13.6.tar.gz",
        ],
        sha256 = "e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20",
        build_file = Label("//third_party:com_github_pybind_pybind11/pybind11.BUILD.bazel"),
        system_build_file = Label("//third_party:com_github_pybind_pybind11/system.BUILD.bazel"),
        # documentation-only
        doc_name = "pybind11",
        doc_homepage = "https://pybind11.readthedocs.io/en/stable/",
        # CMake support is not needed because CMake is not used for the Python build.
    )
