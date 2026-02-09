# Copyright 2026 The TensorStore Authors
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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "mirror_url",
)

def repo():
    maybe(
        http_archive,
        name = "helly25_bzl",
        urls = mirror_url("https://github.com/helly25/bzl/releases/download/0.3.1/bzl-0.3.1.tar.gz"),
        sha256 = "c8e28a3cb7e465b4b71f5d4d366c5796cc0ae822fa510a8adf12cf39a9709902",
        strip_prefix = "bzl-0.3.1",
    )
