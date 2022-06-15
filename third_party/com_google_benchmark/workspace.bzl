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
        name = "com_google_benchmark",
        urls = ["https://github.com/google/benchmark/archive/v1.6.1.zip"],
        sha256 = "367e963b8620080aff8c831e24751852cffd1f74ea40f25d9cc1b667a9dd5e45",
        strip_prefix = "benchmark-1.6.1",
    )

cmake_fetch_content_package(
    name = "com_google_benchmark",
    settings = [
        ("BENCHMARK_ENABLE_TESTING", "OFF"),
        ("BENCHMARK_ENABLE_EXCEPTIONS", "OFF"),
    ],
)

cmake_add_dep_mapping(target_mapping = {
    "@com_google_benchmark//:benchmark": "benchmark::benchmark_main",
    "@com_google_benchmark//:benchmark_main": "benchmark::benchmark_main",
})
