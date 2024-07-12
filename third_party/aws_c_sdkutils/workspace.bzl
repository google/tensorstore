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
load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    maybe(
        third_party_http_archive,
        name = "aws_c_sdkutils",
        sha256 = "4a818563d7c6636b5b245f5d22d4d7c804fa33fc4ea6976e9c296d272f4966d3",
        strip_prefix = "aws-c-sdkutils-0.1.16",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/awslabs/aws-c-sdkutils/archive/refs/tags/v0.1.16.tar.gz",
        ],
        build_file = Label("//third_party:aws_c_sdkutils/aws_c_sdkutils.BUILD.bazel"),
        cmake_name = "aws_c_sdkutils",
        cmake_target_mapping = {
            "@aws_c_sdkutils//:aws_c_sdkutils": "aws_c_sdkutils::aws_c_sdkutils",
        },
        bazel_to_cmake = {},
    )
