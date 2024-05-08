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

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_benchmark",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/google/benchmark/archive/v1.8.3.zip",
        ],
        sha256 = "abfc22e33e3594d0edf8eaddaf4d84a2ffc491ad74b6a7edc6e7a608f690e691",
        strip_prefix = "benchmark-1.8.3",
        patches = [
            Label("//third_party:com_google_benchmark/patches/fix_mingw.diff"),
        ],
        patch_args = ["-p1"],
        cmake_name = "benchmark",
        bazel_to_cmake = {
            "include": [""],
        },
    )
