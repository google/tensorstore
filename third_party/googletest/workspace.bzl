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
load("//third_party:repo.bzl", "third_party_http_archive")

# REPO_BRANCH = main

def repo():
    maybe(
        third_party_http_archive,
        name = "googletest",
        urls = ["https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/google/googletest/archive/cd430b47a54841ec45d64d2377d7cabaf0eba610.zip"],  # main(2025-04-23)
        sha256 = "0970814192d0a48be6ea6626d05569fd8587c54e874323dd0dc85d9ef36c7c86",
        strip_prefix = "googletest-cd430b47a54841ec45d64d2377d7cabaf0eba610",
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
