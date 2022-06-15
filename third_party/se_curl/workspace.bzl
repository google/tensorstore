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

load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//:cmake_helpers.bzl", "cmake_add_dep_mapping", "cmake_find_package")

def repo():
    maybe(
        third_party_http_archive,
        name = "se_curl",
        strip_prefix = "curl-7.83.1",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/curl.se/download/curl-7.83.1.tar.gz",
            "https://curl.se/download/curl-7.83.1.tar.gz",
        ],
        sha256 = "93fb2cd4b880656b4e8589c912a9fd092750166d555166370247f09d18f5d0c0",
        build_file = Label("//third_party:se_curl/bundled.BUILD.bazel"),
        system_build_file = Label("//third_party:se_curl/system.BUILD.bazel"),
    )

cmake_find_package(
    name = "CURL",
    fallback = True,
    settings = [
        ("BUILD_CURL_EXE", "OFF"),
        ("HTTP_ONLY", "ON"),
        ("CURL_BROTLI", "ON"),
        #        ("USE_NGHTTP2", "ON"),  # TODO: Enable this
    ],
)

cmake_add_dep_mapping(target_mapping = {
    "@se_curl//:curl": "CURL::libcurl",
})
