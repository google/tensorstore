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
        strip_prefix = "riegeli-f10b305da3c9b8a36ee2b5eac182b20290035da2",
        urls = [
            "https://github.com/google/riegeli/archive/f10b305da3c9b8a36ee2b5eac182b20290035da2.tar.gz",  # master(2022-10-28)
        ],
        sha256 = "75265d270ae69f38be64fa301f15480fd15200cb070a7b8b0bbf602520325bc3",
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
