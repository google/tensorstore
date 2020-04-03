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
        strip_prefix = "pybind11-2.4.2",
        urls = [
            "https://github.com/pybind/pybind11/archive/v2.4.2.tar.gz",
        ],
        sha256 = "e212e3043cb7a15466abb8896c6924c1ce40ae2988d8c24c111afcb30735fb8f",
        build_file = Label("//third_party:com_github_pybind_pybind11/bundled.BUILD.bazel"),
    )
