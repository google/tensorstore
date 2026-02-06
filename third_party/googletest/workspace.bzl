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

# REPO_BRANCH = main

def repo():
    maybe(
        third_party_http_archive,
        name = "googletest",
        doc_version = "20260206-5a9c3f9",
        urls = mirror_url("https://github.com/google/googletest/archive/5a9c3f9e8d9b90bbbe8feb32902146cb8f7c1757.zip"),  # main(2026-02-06)
        sha256 = "b83b54c3bf02f6e501f85ba99b8f7359c4c19c33d3a69857cdffb9dc501f7724",
        strip_prefix = "googletest-5a9c3f9e8d9b90bbbe8feb32902146cb8f7c1757",
        cmake_name = "GTest",
        bazel_to_cmake = {
            "include": [""],
        },
        cmake_target_mapping = {
            "//:gtest": "GTest::gmock",
            "//:gtest_main": "GTest::gmock_main",
        },
        cmake_aliases = {
            "GTest::gtest": "GTest::gmock",
            "GTest::gtest_main": "GTest::gmock_main",
        },
    )
