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

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_benchmark",
        urls = ["https://github.com/google/benchmark/archive/v1.7.0.zip"],
        sha256 = "e0e6a0f2a5e8971198e5d382507bfe8e4be504797d75bb7aec44b5ea368fa100",
        strip_prefix = "benchmark-1.7.0",
        cmake_name = "benchmark",
        bazel_to_cmake = {
            "include": [""],
        },
    )
