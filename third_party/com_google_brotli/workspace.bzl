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

load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_brotli",
        urls = ["https://github.com/google/brotli/archive/6d03dfbedda1615c4cba1211f8d81735575209c8.zip"],  # master(2022-11-08)
        sha256 = "73a89a4a5ad295eed881795f2767ee9f7542946011f4b30385bcf2caef899df3",
        strip_prefix = "brotli-6d03dfbedda1615c4cba1211f8d81735575209c8",
        system_build_file = Label("//third_party:com_google_brotli/system.BUILD.bazel"),
        cmake_name = "Brotli",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            ":brotlidec": "Brotli::brotlidec",
            ":brotlienc": "Brotli::brotlienc",
        },
    )
