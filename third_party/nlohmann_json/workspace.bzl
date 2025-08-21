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
        name = "nlohmann_json",
        strip_prefix = "json-3.12.0",
        urls = mirror_url("https://github.com/nlohmann/json/archive/v3.12.0.zip"),
        sha256 = "34660b5e9a407195d55e8da705ed26cc6d175ce5a6b1fb957e701fb4d5b04022",
        system_build_file = Label("//third_party:nlohmann_json/system.BUILD.bazel"),
        # documentation-only
        doc_name = "nlohmann/json",
        doc_homepage = "https://json.nlohmann.me/",
        doc_version = "3.12.0",
        cmake_name = "nlohmann_json",
        cmake_target_mapping = {
            "//:json": "nlohmann_json::nlohmann_json",
        },
        bazel_to_cmake = {},
    )
