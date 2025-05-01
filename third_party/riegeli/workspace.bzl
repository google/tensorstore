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
    "third_party_http_archive",
)

# REPO_BRANCH = master

def repo():
    maybe(
        third_party_http_archive,
        name = "riegeli",
        doc_version = "20250415-9da769a3",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/google/riegeli/archive/bcae1d3b1f6da547e77119c463bcab67d7f785d3.tar.gz",  # master(2025-04-29)
        ],
        sha256 = "b45c9b8b3f96b68272b9fa1bc543e7c286717b54a36e877115f39c69a0abf346",
        strip_prefix = "riegeli-bcae1d3b1f6da547e77119c463bcab67d7f785d3",
        patches = [
            # Riegeli patch to build on MSVC
            Label("//third_party:riegeli/patches/chain.diff"),
        ],
        patch_args = ["-p1"],
        repo_mapping = {
            "@net_zstd": "@zstd",
            "@com_google_googleapis": "@googleapis",
            "@com_google_googletest": "@googletest",
            "@com_google_absl": "@abseil-cpp",
        },
        cmake_name = "riegeli",
        bazel_to_cmake = {
            "args": [
                "--exclude-target=//riegeli/digests:highwayhash_digester",
            ],
            "include": ["riegeli/**"],
            "exclude": [
                "riegeli/brotli/**",
                "riegeli/chunk_encoding/**",
                "riegeli/lz4/**",
                "riegeli/records/**",
                "riegeli/snappy/**",
                "riegeli/tensorflow/**",
            ],
        },
    )
