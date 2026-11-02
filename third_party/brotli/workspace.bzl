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
        doc_version = "1.2.1",
        urls = mirror_url("https://github.com/google/brotli/archive/v1.2.1.zip"),
        sha256 = "ca87a5987470399c36752c7468c45df5e3004720625c8f808fe81a32783d8b5b",
        strip_prefix = "brotli-1.2.1",
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
