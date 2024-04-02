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

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)

def repo():
    maybe(
        third_party_http_archive,
        name = "net_zstd",
        strip_prefix = "zstd-1.5.5",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/facebook/zstd/archive/v1.5.5.zip",
        ],
        sha256 = "c5c8daa1d40dabc51790c62a5b86af2b36dfc4e1a738ff10dc4a46ea4e68ee51",
        build_file = Label("//third_party:net_zstd/zstd.BUILD.bazel"),
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
