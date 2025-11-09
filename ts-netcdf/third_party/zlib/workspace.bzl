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

# Using chromium-zlib, source from: https://chromium.googlesource.com/chromium/src/third_party/zlib/

def repo():
    maybe(
        third_party_http_archive,
        name = "zlib",
        # We use the chromium fork of zlib, but generally it tracks here:
        doc_name = "chromium-zlib",
        doc_version = "1.3.1.1-20250425",
        doc_homepage = "https://chromium.googlesource.com/chromium/src/third_party/zlib/",
        # googlesource does not cache archive files; the sha256 is only valid for the mirror.
        sha256 = "eafdc0818f091a50b0ebf439f92bce28fd5f324e93ef8644f7d8fe6c103ddf35",
        urls = mirror_url("https://chromium.googlesource.com/chromium/src/third_party/zlib/+archive/a26c752876190c456e44188db38931561bfa7064.tar.gz"),  # main(2025-04-25)
        build_file = Label("//third_party:zlib/zlib.BUILD.bazel"),
        system_build_file = Label("//third_party:zlib/system.BUILD.bazel"),
        cmake_name = "ZLIB",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "//:zlib": "ZLIB::ZLIB",
        },
        cmake_package_redirect_libraries = {
            "ZLIB": "ZLIB::ZLIB",
        },
    )
