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

# REPO_BRANCH = master

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_riegeli",
        strip_prefix = "riegeli-81c414981883a606ef4865e6b49353d516aa605d",
        urls = [
            "https://github.com/google/riegeli/archive/81c414981883a606ef4865e6b49353d516aa605d.tar.gz",  # master(2023-04-01)
        ],
        sha256 = "d17b323ce1c8bdf73d9947ad13384ce8aa72adce6ead9b66dd4e6dbf2f013e50",
        patches = [
            # Use absl crc32c rather than separate crc32c library.
            "//third_party:com_google_riegeli/patches/absl-crc32c.diff",
        ],
        patch_args = ["-p1"],
        repo_mapping = {
            "@zlib": "@net_zlib",
            "@bzip2": "@org_sourceware_bzip2",
            "@xz": "@org_tukaani_xz",
        },
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
