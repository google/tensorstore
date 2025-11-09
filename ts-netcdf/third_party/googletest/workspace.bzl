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
        urls = mirror_url("https://github.com/google/googletest/archive/a05c0915074bcd1b82f232e081da9bb6c205c28d.zip"),  # main(2025-08-19)
        sha256 = "c05716867701b60562a647d0c9202d6effee53c79e4bc10d4106b3d30fb4482c",
        strip_prefix = "googletest-a05c0915074bcd1b82f232e081da9bb6c205c28d",
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
