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
        strip_prefix = "riegeli-301c122a77aa60fd01396b5b9a7901218494e096",
        urls = [
            "https://github.com/google/riegeli/archive/301c122a77aa60fd01396b5b9a7901218494e096.tar.gz",  # master(2022-11-08)
        ],
        sha256 = "64b8848baef329b382a64e30e1c58708de061bf5a6e017b8599d9ebd116f04d9",
        cmake_name = "riegeli",
        bazel_to_cmake = {
            "include": ["riegeli/**"],
            "exclude": [
                "riegeli/tensorflow/**",
                "riegeli/snappy/**",
                "riegeli/records/**",
                "riegeli/brotli/**",
                "riegeli/chunk_encoding/**",
                "riegeli/lz4/**",
                "riegeli/digests/**",
                "riegeli/bzip2/**",
                "riegeli/zstd/**",
                "riegeli/zlib/**",
            ],
        },
    )
