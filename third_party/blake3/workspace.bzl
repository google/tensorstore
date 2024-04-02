# Copyright 2023 The TensorStore Authors
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
load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)

def repo():
    maybe(
        third_party_http_archive,
        name = "blake3",
        strip_prefix = "BLAKE3-64747d48ffe9d1fbf4b71e94cabeb8a211461081",
        sha256 = "4d467f68a61a3ce4fd83b1b00cb7698cb58a3b6ec1635b3e6d5602f1413f28df",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/BLAKE3-team/blake3/archive/64747d48ffe9d1fbf4b71e94cabeb8a211461081.tar.gz",  # master(2023-03-01)
        ],
        build_file = Label("//third_party:blake3/blake3.BUILD.bazel"),
        system_build_file = Label("//third_party:blake3/system.BUILD.bazel"),
        cmake_name = "BLAKE3",
        cmake_languages = ["ASM", "ASM_MASM"],
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "@blake3//:blake3": "BLAKE3::BLAKE3",
        },
    )
