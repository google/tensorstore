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

def repo():
    maybe(
        third_party_http_archive,
        name = "google_benchmark",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/google/benchmark/archive/v1.9.2.zip",
        ],
        sha256 = "a13cdcd3b0e3725e371f977a2855fecd651fc0236c67da16e325b726057b15b4",
        strip_prefix = "benchmark-1.9.2",
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
