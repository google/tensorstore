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

load("//third_party:repo.bzl", "third_party_http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//:cmake_helpers.bzl", "cmake_add_dep_mapping", "cmake_fetch_content_package")

# REPO_BRANCH = master

def repo():
    maybe(
        third_party_http_archive,
        name = "com_github_pybind_pybind11",
        strip_prefix = "pybind11-2.9.2",
        urls = [
            "https://github.com/pybind/pybind11/archive/v2.9.2.tar.gz",
        ],
        sha256 = "6bd528c4dbe2276635dc787b6b1f2e5316cf6b49ee3e150264e455a0d68d19c1",
        build_file = Label("//third_party:com_github_pybind_pybind11/bundled.BUILD.bazel"),
    )

cmake_fetch_content_package(
    name = "com_github_pybind_pybind11",
)

cmake_add_dep_mapping(target_mapping = {
    "@com_github_pybind_pybind11//:pybind11": "pybind11::module",
})
