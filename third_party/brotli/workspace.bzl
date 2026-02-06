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
        doc_version = "1.1.0-20260206-5fa73e2",
        urls = mirror_url("https://github.com/google/brotli/archive/5fa73e23bee34f84148719576a7a434f0fc43dc8.zip"),  # master(2026-02-06)
        sha256 = "f45ded02b61979183039ee66ec29c52bc44827a71676732df57821c40e33bba9",
        strip_prefix = "brotli-5fa73e23bee34f84148719576a7a434f0fc43dc8",
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
