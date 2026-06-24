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
        name = "brotli",
        doc_version = "1.1.0-20260620-83fe766",
        urls = mirror_url("https://github.com/google/brotli/archive/83fe766bc81f7911a78716b8d3b3d01367009995.zip"),  # master(2026-06-20)
        sha256 = "7062dad70a692e5dd2abe3cb9ea65f379553f6727b2b7125d05b57414f46370a",
        strip_prefix = "brotli-83fe766bc81f7911a78716b8d3b3d01367009995",
        patches = [
            Label("//third_party:brotli/patches/fix_ror.diff"),
        ],
        patch_args = ["-p1"],
        repo_mapping = {
            "@org_brotli": "@brotli",
        },
        system_build_file = Label("//third_party:brotli/system.BUILD.bazel"),
        cmake_name = "Brotli",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "//:brotlidec": "Brotli::brotlidec",
            "//:brotlienc": "Brotli::brotlienc",
        },
    )
