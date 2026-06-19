# Copyright 2021 The TensorStore Authors
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
        name = "bazel_features",
        sha256 = "89eca73d4c334cf664f84920365d2ce04e2c98099b89f7c5b676b5f377c8e754",
        strip_prefix = "bazel_features-1.48.1",
        urls = mirror_url("https://github.com/bazel-contrib/bazel_features/archive/v1.48.1.tar.gz"),
    )
