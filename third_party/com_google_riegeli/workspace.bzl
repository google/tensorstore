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
        strip_prefix = "riegeli-4512ad845a3d678e21a014af4f1d4f6fa910e6b4",
        urls = [
            "https://github.com/google/riegeli/archive/4512ad845a3d678e21a014af4f1d4f6fa910e6b4.tar.gz",  # master(2022-12-08)
        ],
        sha256 = "9050366b283a9c2cadc828cd702b01ed76e7055b2d21d452358aec93d4304333",
        cmake_name = "riegeli",
        bazel_to_cmake = {
            "include": ["riegeli/**"],
            "exclude": [
                "riegeli/brotli/**",
                "riegeli/bzip2/**",
                "riegeli/chunk_encoding/**",
                "riegeli/digests/**",
                "riegeli/lz4/**",
                "riegeli/records/**",
                "riegeli/snappy/**",
                "riegeli/tensorflow/**",
                "riegeli/zlib/**",
                "riegeli/zstd/**",
            ],
        },
    )
