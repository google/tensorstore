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
load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    maybe(
        third_party_http_archive,
        name = "aws_c_sdkutils",
        sha256 = "66bd7a8679703386aec1539407aaed0942a78032fe340ab44e810a3cf6d7e505",
        strip_prefix = "aws-c-sdkutils-0.1.19",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/awslabs/aws-c-sdkutils/archive/v0.1.19.tar.gz",
        ],
        build_file = Label("//third_party:aws_c_sdkutils/aws_c_sdkutils.BUILD.bazel"),
        cmake_name = "aws_c_sdkutils",
        cmake_target_mapping = {
            "@aws_c_sdkutils//:aws_c_sdkutils": "aws_c_sdkutils::aws_c_sdkutils",
        },
        bazel_to_cmake = {},
    )
