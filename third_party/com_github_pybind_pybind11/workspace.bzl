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

def repo():
    maybe(
        third_party_http_archive,
        name = "com_github_pybind_pybind11",
        strip_prefix = "pybind11-d58699c9ab9d20010b15aa38382ad517a8280179",
        urls = [
            "https://github.com/pybind/pybind11/archive/d58699c9ab9d20010b15aa38382ad517a8280179.tar.gz",  # master(2021-09-24)
        ],
        sha256 = "ddf32c7ceb3d7deae98f04c85c5607a6d3092f809a686ee72b1b3c528ff27b75",
        build_file = Label("//third_party:com_github_pybind_pybind11/bundled.BUILD.bazel"),
    )
