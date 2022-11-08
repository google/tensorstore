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

# REPO_BRANCH = main

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_googletest",
        urls = ["https://github.com/google/googletest/archive/6b63c98ac43efc992122f1da12aaf0a0e0658206.zip"],  # main(2022-11-08)
        sha256 = "133d8d8f78a6a74d974fc31050b7e2e6af07fde013a0bf3bb517d1e63d55b2ff",
        strip_prefix = "googletest-6b63c98ac43efc992122f1da12aaf0a0e0658206",
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
