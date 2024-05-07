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
        strip_prefix = "utf8_range-1d1ea7e3fedf482d4a12b473c1ed25fe0f371a45",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/protocolbuffers/utf8_range/archive/1d1ea7e3fedf482d4a12b473c1ed25fe0f371a45.zip",  # main(2024-05-03)
        ],
        sha256 = "bf4798b9635a3b34e40a30cb54b5bc185807945da8839e041ab8d91f268d26b1",
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
