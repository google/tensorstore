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

# Version 1.24.5 is the last version that supports WORKSPACE builds.
def repo():
    maybe(
        http_archive,
        name = "build_bazel_apple_support",
        urls = mirror_url("https://github.com/bazelbuild/apple_support/archive/1.24.5.tar.gz"),
        sha256 = "e5179535c56801ab43379deb6acfbca17da65f0b05d95e9b7565d2ae1cd22411",
        strip_prefix = "apple_support-1.24.5",
    )
