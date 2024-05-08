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
        name = "com_github_nlohmann_json",
        strip_prefix = "json-3.11.3",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/nlohmann/json/archive/v3.11.3.zip",
        ],
        sha256 = "04022b05d806eb5ff73023c280b68697d12b93e1b7267a0b22a1a39ec7578069",
        system_build_file = Label("//third_party:com_github_nlohmann_json/system.BUILD.bazel"),
        # documentation-only
        doc_name = "nlohmann/json",
        doc_homepage = "https://json.nlohmann.me/",
        doc_version = "3.11.3",
        cmake_name = "nlohmann_json",
        cmake_target_mapping = {
            "@com_github_nlohmann_json//:json": "nlohmann_json::nlohmann_json",
        },
        bazel_to_cmake = {},
    )
