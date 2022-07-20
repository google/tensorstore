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
        urls = ["https://github.com/google/googletest/archive/7735334a46da480a749945c0f645155d90d73855.zip"],  # main(2022-07-20)
        sha256 = "cd2d4721e0cfa5f573c1cd6794ce9401b8a3b418cf734e51e94093c085e3535a",
        strip_prefix = "googletest-7735334a46da480a749945c0f645155d90d73855",
        repo_mapping = {
            "@com_googlesource_code_re2": "@com_google_re2",
        },
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
