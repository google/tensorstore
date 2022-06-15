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
load("//:cmake_helpers.bzl", "cmake_add_dep_mapping", "cmake_find_package", "cmake_raw", "cmake_set_section")

# REPO_BRANCH = main

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_googletest",
        urls = ["https://github.com/google/googletest/archive/191ca1f3a9262b90a586ae2c2e8c742c3d867801.zip"],  # main(2022-06-14)
        sha256 = "d0b0b8fffa58489f0803b230d2971bd6344fc3a0068f352fd7075bca8ea6c4cc",
        strip_prefix = "googletest-191ca1f3a9262b90a586ae2c2e8c742c3d867801",
    )

cmake_set_section(section = 100)

cmake_add_dep_mapping(target_mapping = {
    "@com_google_googletest//:gtest": "GTest::gtest_main",
    "@com_google_googletest//:gtest_main": "GTest::gtest_main",
})

cmake_find_package(
    name = "GTest",
    fallback = True,
)

cmake_raw(
    text = """

check_target(GTest::gtest)
check_target(GTest::gtest_main)
check_target(GTest::gmock)
check_target(GTest::gmock_main)

""",
)
