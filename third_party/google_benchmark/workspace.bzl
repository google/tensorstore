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

def repo():
    maybe(
        third_party_http_archive,
        name = "google_benchmark",
        urls = mirror_url("https://github.com/google/benchmark/archive/v1.9.5.zip"),
        sha256 = "68c9c65cee4864db42c3af9ff5b5cfa32ce1b01d9653136c5f4ff96e18a9b8f2",
        strip_prefix = "benchmark-1.9.5",
        patches = [
            Label("//third_party:google_benchmark/patches/fix_mingw.diff"),
        ],
        doc_version = "1.9.0",
        patch_args = ["-p1"],
        cmake_name = "benchmark",
        bazel_to_cmake = {
            "include": [""],
        },
    )
