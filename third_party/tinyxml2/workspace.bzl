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

# buildifier: disable=module-docstring

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)

def repo():
    maybe(
        third_party_http_archive,
        name = "tinyxml2",
        strip_prefix = "tinyxml2-11.0.0",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/leethomason/tinyxml2/archive/11.0.0.tar.gz",
        ],
        sha256 = "5556deb5081fb246ee92afae73efd943c889cef0cafea92b0b82422d6a18f289",
        build_file = Label("//third_party:tinyxml2/tinyxml2.BUILD.bazel"),
        system_build_file = Label("//third_party:tinyxml2/system.BUILD.bazel"),
        cmake_name = "tinyxml2",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "@tinyxml2//:tinyxml2": "tinyxml2::tinyxml2",
        },
    )
