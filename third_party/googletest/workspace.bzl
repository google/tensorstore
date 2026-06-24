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
        doc_version = "20260620-0b1e895",
        urls = mirror_url("https://github.com/google/googletest/archive/0b1e895ba4226c2fda5ee0178c9b5b1195a741aa.zip"),  # main(2026-06-20)
        sha256 = "3fcc9a790e7253b240cdabb82eedcf2058159cdbdb151c62591a07a99d2a4709",
        strip_prefix = "googletest-0b1e895ba4226c2fda5ee0178c9b5b1195a741aa",
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
