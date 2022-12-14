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
        name = "net_zstd",
        strip_prefix = "zstd-1.5.2",
        urls = [
            "https://github.com/facebook/zstd/archive/v1.5.2.zip",
        ],
        sha256 = "53f4696f3cec8703f12d3402707a6aaf7eb92d43c90d61e1d32454bda5da7b9c",
        build_file = Label("//third_party:net_zstd/bundled.BUILD.bazel"),
        system_build_file = Label("//third_party:net_zstd/system.BUILD.bazel"),
        cmake_name = "Zstd",
        bazel_to_cmake = {},
        cmake_languages = ["C", "ASM"],
        cmake_target_mapping = {
            ":zstdlib": "Zstd::Zstd",
        },
        cmake_package_redirect_libraries = {
            "Zstd": "Zstd::Zstd",
        },
        # NOTE: Verify if this helps: https://github.com/google/tensorstore/issues/69
        # cmakelists_prefix = 'SET(CMAKE_ASM_COMPILER "${CMAKE_C_COMPILER}")\n',
    )
