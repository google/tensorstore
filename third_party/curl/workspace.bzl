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
load(
    "//third_party:repo.bzl",
    "mirror_url",
    "third_party_http_archive",
)

def repo():
    maybe(
        third_party_http_archive,
        name = "curl",
        doc_name = "curl",
        doc_version = "8.18.0",
        doc_homepage = "https://curl.se/",
        strip_prefix = "curl-8.18.0",
        urls = mirror_url("https://curl.se/download/curl-8.18.0.tar.gz"),
        sha256 = "e9274a5f8ab5271c0e0e6762d2fce194d5f98acc568e4ce816845b2dcc0cf88f",
        build_file = Label("//third_party:curl/curl.BUILD.bazel"),
        system_build_file = Label("//third_party:curl/system.BUILD.bazel"),
        cmake_name = "CURL",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "//:curl": "CURL::libcurl",
        },
    )
