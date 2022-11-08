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

load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        third_party_http_archive,
        name = "org_lz4",
        strip_prefix = "lz4-1.9.4",
        urls = [
            "https://github.com/lz4/lz4/archive/v1.9.4.zip",
        ],
        sha256 = "37e63d56fb9cbe2e430c7f737a404cd4b98637b05e1467459d5c8fe1a4364cc3",
        build_file = Label("//third_party:org_lz4/bundled.BUILD.bazel"),
        system_build_file = Label("//third_party:org_lz4/system.BUILD.bazel"),
        cmake_name = "LZ4",
        cmake_target_mapping = {
            "@org_lz4//:lz4": "LZ4::LZ4",
        },
        bazel_to_cmake = {},
        cmake_package_redirect_libraries = {
            "LZ4": "LZ4::LZ4",
        },
    )
