# Copyright 2023 The TensorStore Authors
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

def repo():
    maybe(
        http_archive,
        name = "build_bazel_apple_support",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/bazelbuild/apple_support/releases/download/1.17.1/apple_support.1.17.1.tar.gz",
        ],
        sha256 = "b53f6491e742549f13866628ddffcc75d1f3b2d6987dc4f14a16b242113c890b",
    )
