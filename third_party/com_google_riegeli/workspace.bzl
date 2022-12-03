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
        strip_prefix = "riegeli-b235d1077b35fd0dbcb212f05a3c8ae89b4784ff",
        urls = [
            "https://github.com/google/riegeli/archive/b235d1077b35fd0dbcb212f05a3c8ae89b4784ff.tar.gz",  # master(2022-12-01)
        ],
        sha256 = "21227d8f3750b95bf44a5a448991edba2b9fe42ef4e1e51dc00229a25ff75101",
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
