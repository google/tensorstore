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
        name = "rules_proto",
        strip_prefix = "rules_proto-6.0.0-rc0",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/bazelbuild/rules_proto/archive/6.0.0-rc0.tar.gz",
        ],
        sha256 = "903af49528dc37ad2adbb744b317da520f133bc1cbbecbdd2a6c546c9ead080b",
    )
