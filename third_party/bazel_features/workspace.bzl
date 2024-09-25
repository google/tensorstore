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

def repo():
    maybe(
        http_archive,
        name = "bazel_features",
        sha256 = "bdc12fcbe6076180d835c9dd5b3685d509966191760a0eb10b276025fcb76158",
        strip_prefix = "bazel_features-1.17.0",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/bazel-contrib/bazel_features/archive/v1.17.0.tar.gz",
        ],
    )
