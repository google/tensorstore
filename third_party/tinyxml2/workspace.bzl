# Copyright 2024 The TensorStore Authors
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
        name = "tinyxml2",
        strip_prefix = "tinyxml2-10.0.0",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/leethomason/tinyxml2/archive/10.0.0.tar.gz",
        ],
        sha256 = "3bdf15128ba16686e69bce256cc468e76c7b94ff2c7f391cc5ec09e40bff3839",
        build_file = Label("//third_party:tinyxml2/tinyxml2.BUILD.bazel"),
        system_build_file = Label("//third_party:tinyxml2/system.BUILD.bazel"),
        cmake_name = "tinyxml2",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "@tinyxml2//:tinyxml2": "tinyxml2::tinyxml2",
        },
    )
