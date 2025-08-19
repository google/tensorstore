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

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "mirror_url",
    "third_party_http_archive",
)

def repo():
    maybe(
        third_party_http_archive,
        name = "libyuv",
        doc_name = "libyuv",
        doc_homepage = "https://chromium.googlesource.com/libyuv/libyuv/",
        urls = mirror_url("https://chromium.googlesource.com/libyuv/libyuv/+archive/1e40e34573c3861480d107cd4a4ce290df79951f.tar.gz"),  # main(2025-04-26)
        # googlesource does not cache archive files; the sha256 is only valid for the mirror.
        sha256 = "dc032a4462e6ce994c2c0f01d07a8742f0355661d4f9716ddb621f4578a4f4fc",
        build_file = Label("//third_party:libyuv/libyuv.BUILD.bazel"),
        cmake_name = "libyuv",
        cmake_target_mapping = {
            "//:libyuv": "LIBYUV::LIBYUV",
        },
        bazel_to_cmake = {},
        cmake_package_redirect_libraries = {
            "LIBYUV": "LIBYUV::LIBYUV",
        },
    )
