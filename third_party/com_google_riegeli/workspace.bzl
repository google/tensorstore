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
        strip_prefix = "riegeli-8bfa2cb12715cad7b7f65fa8122f41b5ab5a7103",
        urls = [
            "https://github.com/google/riegeli/archive/8bfa2cb12715cad7b7f65fa8122f41b5ab5a7103.tar.gz",  # master(2023-03-14)
        ],
        sha256 = "3d90afaf477e782fc993394d3f4036d1323772e3745baf45b8674a0da07f8469",
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
                "riegeli/digests/**",
                "riegeli/lz4/**",
                "riegeli/records/**",
                "riegeli/snappy/**",
                "riegeli/tensorflow/**",
            ],
        },
    )
