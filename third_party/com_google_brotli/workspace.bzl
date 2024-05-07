# Copyright 2021 The TensorStore Authors
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
        name = "com_google_brotli",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/google/brotli/archive/d01a4caaa80c0072fe1b6bf073814b9400667fcc.zip",  # master(2024-05-02)
        ],
        sha256 = "45b86686a273c5f2a5fd14419be60d50a3ea86e98243d28faba2c74b7b2f01f9",
        strip_prefix = "brotli-d01a4caaa80c0072fe1b6bf073814b9400667fcc",
        system_build_file = Label("//third_party:com_google_brotli/system.BUILD.bazel"),
        cmake_name = "Brotli",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            ":brotlidec": "Brotli::brotlidec",
            ":brotlienc": "Brotli::brotlienc",
        },
    )
