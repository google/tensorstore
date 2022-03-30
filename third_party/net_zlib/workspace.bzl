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
        name = "net_zlib",
        sha256 = "91844808532e5ce316b3c010929493c0244f3d37593afd6de04f71821d5136d9",
        strip_prefix = "zlib-1.2.12",
        urls = [
            "https://zlib.net/zlib-1.2.12.tar.gz",
        ],
        build_file = Label("//third_party:net_zlib/bundled.BUILD.bazel"),
        system_build_file = Label("//third_party:net_zlib/system.BUILD.bazel"),
    )

cmake_find_package(name = "ZLIB")

cmake_add_dep_mapping(target_mapping = {
    "@net_zlib//:zlib": "ZLIB::ZLIB",
})
