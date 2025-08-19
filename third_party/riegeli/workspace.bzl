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
        doc_version = "20250607-c4d1f27",
        urls = mirror_url("https://github.com/google/riegeli/archive/c4d1f275ed44db839385e494c3a969ae232d6e10.tar.gz"),  # master(2025-06-07)
        sha256 = "0ca5be90a2e184fc2a9590e804c476a004f04b48c40ca488a6b608fcd224b32f",
        strip_prefix = "riegeli-c4d1f275ed44db839385e494c3a969ae232d6e10",
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
