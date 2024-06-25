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
load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_snappy",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/google/snappy/archive/52820ea9c6cb0af6ebe4920f806dbfbb0de9eaab.tar.gz",  # main(2024-04-10)
        ],
        sha256 = "eb9aa0369f9e74dfa28c64a03bd3c6663a17a9ffc9e2d80e94f0a3f0f2a37342",
        strip_prefix = "snappy-52820ea9c6cb0af6ebe4920f806dbfbb0de9eaab",
        build_file = Label("//third_party:com_google_snappy/snappy.BUILD.bazel"),
        system_build_file = Label("//third_party:com_google_snappy/system.BUILD.bazel"),
        cmake_name = "Snappy",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "//:snappy": "Snappy::Snappy",
        },
        cmake_package_redirect_libraries = {
            "Snappy": "Snappy::Snappy",
        },
    )
