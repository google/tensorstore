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

# buildifier: disable=module-docstring

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    maybe(
        third_party_http_archive,
        name = "se_curl",
        strip_prefix = "curl-8.10.1",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/curl.se/download/curl-8.10.1.tar.gz",
            "https://curl.se/download/curl-8.10.1.tar.gz",
        ],
        sha256 = "d15ebab765d793e2e96db090f0e172d127859d78ca6f6391d7eafecfd894bbc0",
        build_file = Label("//third_party:se_curl/curl.BUILD.bazel"),
        system_build_file = Label("//third_party:se_curl/system.BUILD.bazel"),
        cmake_name = "CURL",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "@se_curl//:curl": "CURL::libcurl",
        },
    )
