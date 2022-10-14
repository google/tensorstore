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

load("//third_party:repo.bzl", "third_party_http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_github_nlohmann_json",
        urls = [
            "https://github.com/nlohmann/json/releases/download/v3.10.5/include.zip",
        ],
        sha256 = "b94997df68856753b72f0d7a3703b7d484d4745c567f3584ef97c96c25a5798e",
        build_file = Label("//third_party:com_github_nlohmann_json/bundled.BUILD.bazel"),
        system_build_file = Label("//third_party:com_github_nlohmann_json/system.BUILD.bazel"),
        # documentation-only
        doc_name = "nlohmann/json",
        doc_homepage = "https://json.nlohmann.me/",
        doc_version = "3.10.5",
        cmake_name = "nlohmann_json",
        cmake_target_mapping = {
            "@com_github_nlohmann_json//:nlohmann_json": "nlohmann_json::nlohmann_json",
        },
        bazel_to_cmake = {},
    )
