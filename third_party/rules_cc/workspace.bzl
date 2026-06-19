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
load(
    "//third_party:repo.bzl",
    "mirror_url",
)

def repo():
    maybe(
        http_archive,
        name = "rules_cc",
        urls = mirror_url("https://github.com/bazelbuild/rules_cc/archive/0.2.19.tar.gz"),
        strip_prefix = "rules_cc-0.2.19",
        sha256 = "351248f6be41d18694d4d7c390aaebd9f865eea72a4758b2c9d782ae744c97f4",
        repo_mapping = {
            "@protobuf": "@com_google_protobuf",
        },
    )
