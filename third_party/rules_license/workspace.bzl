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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        http_archive,
        name = "rules_license",
        strip_prefix = "rules_license-0.0.7",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/bazelbuild/rules_license/archive/0.0.7.tar.gz",
        ],
        sha256 = "7626bea5473d3b11d44269c5b510a210f11a78bca1ed639b0f846af955b0fe31",
    )
