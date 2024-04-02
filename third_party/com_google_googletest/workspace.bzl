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

# REPO_BRANCH = main

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_googletest",
        urls = ["https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/google/googletest/archive/a798c2f10200b6293a5cc236b5f41b26c1ae7378.zip"],  # main(2023-03-07)
        sha256 = "65789e4fd5bb28ca6a87667ca9f7eea923f82c8c7e48d6677e06b74504720c84",
        strip_prefix = "googletest-a798c2f10200b6293a5cc236b5f41b26c1ae7378",
        repo_mapping = {
            "@com_googlesource_code_re2": "@com_google_re2",
        },
        cmake_name = "GTest",
        bazel_to_cmake = {
            "include": [""],
        },
        cmake_target_mapping = {
            "@com_google_googletest//:gtest": "GTest::gmock",
            "@com_google_googletest//:gtest_main": "GTest::gmock_main",
        },
        cmake_aliases = {
            "GTest::gtest": "GTest::gmock",
            "GTest::gtest_main": "GTest::gmock_main",
        },
    )
