# Copyright 2022 The TensorStore Authors
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

def repo():
    maybe(
        third_party_http_archive,
        name = "libwebp",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/webmproject/libwebp/archive/refs/tags/v1.2.4.tar.gz",
            "https://github.com/webmproject/libwebp/archive/v1.2.4.tar.gz",
        ],
        sha256 = "dfe7bff3390cd4958da11e760b65318f0a48c32913e4d5bc5e8d55abaaa2d32e",
        strip_prefix = "libwebp-1.2.4",
        build_file = Label("//third_party:libwebp/libwebp.BUILD.bazel"),
        system_build_file = Label("//third_party:libwebp/system.BUILD.bazel"),
        cmake_name = "WebP",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "@libwebp//:webp": "WebP::webp",
            "@libwebp//:sharpyuv": "WebP::sharpyuv",
        },
        cmake_package_redirect_libraries = {
            "WebP": "WebP::webp",
            "LIBSHARPYUV": "WebP::sharpyuv",
        },
        cmake_package_aliases = ["libsharpyuv"],
    )
