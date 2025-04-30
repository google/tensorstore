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
        strip_prefix = "riegeli-e54e4300f61cc113e5da816952d20ae070fe7219",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/google/riegeli/archive/e54e4300f61cc113e5da816952d20ae070fe7219.tar.gz",  # master(2023-12-12)
        ],
        sha256 = "c2a40a1a422fe513faa198b325865022ccfbb1ca6317ef5f51a8b60d7c2bdc4c",
        doc_version = "20231212-e54e430",
        patches = [
            # Use absl crc32c rather than separate crc32c library.
            Label("//third_party:riegeli/patches/absl-crc32c.diff"),
            # zstd, bzip2, xz library names have changed.
            Label("//third_party:riegeli/patches/build.diff"),
        ],
        repo_mapping = {
            "@net_zstd": "@zstd",
            "@com_google_googleapis": "@googleapis",
            "@com_google_googletest": "@googletest",
            "@com_google_absl": "@abseil-cpp",
        },
        patch_args = ["-p1"],
        cmake_name = "riegeli",
        bazel_to_cmake = {
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
