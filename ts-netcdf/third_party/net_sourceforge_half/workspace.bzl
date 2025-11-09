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

# buildifier: disable=module-docstring

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "mirror_url",
    "third_party_http_archive",
)

def repo():
    maybe(
        third_party_http_archive,
        name = "net_sourceforge_half",
        urls = mirror_url("https://sourceforge.net/projects/half/files/half/2.2.1/half-2.2.1.zip"),
        sha256 = "76ddbf406e9d9b772ec73af2bf925b38b290b4390cc4064720a08d4b4bca0aa9",
        build_file = Label("//third_party:net_sourceforge_half/half.BUILD.bazel"),
        cmake_name = "half",
        cmake_target_mapping = {
            "@net_sourceforge_half//:half": "half::half",
        },
        bazel_to_cmake = {},
    )
