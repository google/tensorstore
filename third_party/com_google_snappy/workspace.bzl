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
        name = "com_google_snappy",
        urls = ["https://github.com/google/snappy/archive/1.1.9.zip"],
        sha256 = "e170ce0def2c71d0403f5cda61d6e2743373f9480124bcfcd0fa9b3299d428d9",
        strip_prefix = "snappy-1.1.9",
        build_file = Label("//third_party:com_google_snappy/bundled.BUILD.bazel"),
        system_build_file = Label("//third_party:com_google_snappy/system.BUILD.bazel"),
        patches = [
            # https://github.com/google/snappy/pull/142
            "//third_party:com_google_snappy/patches/asm-older-clang-fix.diff",
        ],
        patch_args = ["-p1"],
    )

# https://github.com/google/snappy/blob/main/CMakeLists.txt
# Snappy is only used by blosc
#
# load("//:cmake_helpers.bzl", "cmake_fetch_content_package")
# cmake_fetch_content_package(
#    name = "com_google_snappy",
#    settings = [
#        ("SNAPPY_BUILD_TESTS", "OFF"),
#        ("SNAPPY_BUILD_BENCHMARKS", "OFF"),
#        ("SNAPPY_INSTALL", "OFF"),
#    ],
#)
