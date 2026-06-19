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
        name = "rules_pkg",
        strip_prefix = "rules_pkg-1.3",
        urls = mirror_url("https://github.com/bazelbuild/rules_pkg/archive/1.3.tar.gz"),
        sha256 = "30ab19209d73529cff2924d664e59b20f7a88039f36a289a9e99126a3ff5f8ff",
    )
