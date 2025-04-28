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
        name = "snappy",
        doc_version = "1.2.1-20240817-32ded45",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/google/snappy/archive/6af9287fbdb913f0794d0148c6aa43b58e63c8e3.tar.gz",  # main(2025-04-26)
        ],
        sha256 = "31c566ed653abfa68963232e52a97b9b89841d5255dad3c88bd9d74db3039d28",
        strip_prefix = "snappy-6af9287fbdb913f0794d0148c6aa43b58e63c8e3",
        build_file = Label("//third_party:snappy/snappy.BUILD.bazel"),
        system_build_file = Label("//third_party:snappy/system.BUILD.bazel"),
        cmake_name = "Snappy",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "//:snappy": "Snappy::Snappy",
        },
        cmake_package_redirect_libraries = {
            "Snappy": "Snappy::Snappy",
        },
    )
