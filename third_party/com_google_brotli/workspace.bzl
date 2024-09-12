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
load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_brotli",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/google/brotli/archive/39bcecf4559f9b0e75c6248a75d9c0db6b123d88.zip",  # master(2024-09-10)
        ],
        sha256 = "6c72b4d43cd11c8cfd966dbe293cd7f11a3e4e2e08408861b49198dc50b447c3",
        strip_prefix = "brotli-39bcecf4559f9b0e75c6248a75d9c0db6b123d88",
        patches = [
            Label("//third_party:com_google_brotli/patches/fix_ror.diff"),
        ],
        patch_args = ["-p1"],
        system_build_file = Label("//third_party:com_google_brotli/system.BUILD.bazel"),
        cmake_name = "Brotli",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            ":brotlidec": "Brotli::brotlidec",
            ":brotlienc": "Brotli::brotlienc",
        },
    )
