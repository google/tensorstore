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
load("//:cmake_helpers.bzl", "cmake_add_dep_mapping", "cmake_fetch_content_package")

def repo():
    maybe(
        third_party_http_archive,
        name = "org_blosc_cblosc",
        strip_prefix = "c-blosc-1.21.1",
        urls = [
            "https://github.com/Blosc/c-blosc/archive/v1.21.1.zip",
        ],
        sha256 = "abdf8ad8e5f8a876d67b38d16ff0c40c0456cdce1dcbafe58b589671ff55d31a",
        build_file = Label("//third_party:org_blosc_cblosc/bundled.BUILD.bazel"),
        system_build_file = Label("//third_party:org_blosc_cblosc/system.BUILD.bazel"),
    )

cmake_add_dep_mapping(target_mapping = {
    "@org_blosc_cblosc//:blosc": "blosc",
})

cmake_fetch_content_package(
    name = "org_blosc_cblosc",
    settings = [
        ("BUILD_SHARED", "OFF"),
        ("BUILD_TESTS", "OFF"),
        ("BUILD_FUZZERS", "OFF"),
        ("BUILD_BENCHMARKS", "OFF"),
        ("DEACTIVATE_SNAPPY", "OFF"),
        ("TEST_INCLUDE_BENCH_SHUFFLE_1", "OFF"),
        ("TEST_INCLUDE_BENCH_SHUFFLE_N", "OFF"),
        ("TEST_INCLUDE_BENCH_BITSHUFFLE_1", "OFF"),
        ("TEST_INCLUDE_BENCH_BITSHUFFLE_N", "OFF"),
        ("SNAPPY_BUILD_TESTS", "OFF"),
        ("SNAPPY_BUILD_BENCHMARKS", "OFF"),
    ],
)
