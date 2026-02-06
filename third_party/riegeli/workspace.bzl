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

# REPO_BRANCH = master

def repo():
    maybe(
        third_party_http_archive,
        name = "riegeli",
        doc_version = "20260206-43b1456",
        urls = mirror_url("https://github.com/google/riegeli/archive/43b1456a2341b18e3506ad3c01c0f6bf7e37221d.tar.gz"),  # master(2026-02-06)
        sha256 = "abc1b8a081329395b0968441e66a154bd96bcc64c1eaadb55539bddeb65ca878",
        strip_prefix = "riegeli-43b1456a2341b18e3506ad3c01c0f6bf7e37221d",
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
                "riegeli/gcs/**",
            ],
        },
    )
