# Copyright 2020 The TensorStore Authors
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

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_protobuf_utf8_range",
        strip_prefix = "utf8_range-cd1507d1479815fbcd8ff24dc05a978a62098bae",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/protocolbuffers/utf8_range/archive/cd1507d1479815fbcd8ff24dc05a978a62098bae.zip",  # main(2023-05-24)
        ],
        sha256 = "707b7278bb002ac1901f4afb13b51333b3a3aefd880616aa2587eda10a8a0824",
        cmake_name = "utf8_range",
        cmake_target_mapping = {
            "//:utf8_range": "utf8_range::utf8_range",
            "//:utf8_validity": "utf8_range::utf8_validity",
        },
        bazel_to_cmake = {
            "exclude": [
                "fuzz/**",
            ],
        },
    )
