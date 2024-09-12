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
load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_snappy",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/google/snappy/archive/32ded457c0b1fe78ceb8397632c416568d6714a0.tar.gz",  # main(2024-09-10)
        ],
        sha256 = "677d1dd8172bac1862e6c8d7bbe1fe9fb2320cfd11ee04756b1ef8b3699c6135",
        strip_prefix = "snappy-32ded457c0b1fe78ceb8397632c416568d6714a0",
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
