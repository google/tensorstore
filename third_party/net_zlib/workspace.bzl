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

def repo():
    maybe(
        third_party_http_archive,
        name = "net_zlib",
        sha256 = "b3a24de97a8fdbc835b9833169501030b8977031bcb54b3b3ac13740f846ab30",
        strip_prefix = "zlib-1.2.13",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/zlib.net/zlib-1.2.13.tar.gz",
        ],
        build_file = Label("//third_party:net_zlib/bundled.BUILD.bazel"),
        system_build_file = Label("//third_party:net_zlib/system.BUILD.bazel"),
        cmake_name = "ZLIB",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "@net_zlib//:zlib": "ZLIB::ZLIB",
        },
        cmake_package_redirect_libraries = {
            "ZLIB": "ZLIB::ZLIB",
        },
    )
