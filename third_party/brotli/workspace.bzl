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
        doc_version = "1.1.0-20250426-440e036",
        urls = mirror_url("https://github.com/google/brotli/archive/440e03642b891968a76b6d088d70f01f06e0c349.zip"),  # master(2025-04-26)
        sha256 = "1c8d49d72f5cb1ca892ca4bc38021c7bd94a74f10cc493b6756b8dd550701189",
        strip_prefix = "brotli-440e03642b891968a76b6d088d70f01f06e0c349",
        patches = [
            Label("//third_party:brotli/patches/fix_ror.diff"),
        ],
        patch_args = ["-p1"],
        system_build_file = Label("//third_party:brotli/system.BUILD.bazel"),
        cmake_name = "Brotli",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "//:brotlidec": "Brotli::brotlidec",
            "//:brotlienc": "Brotli::brotlienc",
        },
    )
