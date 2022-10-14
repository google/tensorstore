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
        name = "com_google_snappy",
        urls = ["https://github.com/google/snappy/archive/984b191f0fefdeb17050b42a90b7625999c13b8d.tar.gz"],  # main(2022-10-12)
        sha256 = "2e458b7017cd58dcf1469ab315389e85e7f445bd035188f2983f81fb19ecfb29",
        strip_prefix = "snappy-984b191f0fefdeb17050b42a90b7625999c13b8d",
        build_file = Label("//third_party:com_google_snappy/bundled.BUILD.bazel"),
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
