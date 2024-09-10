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
        name = "aws_c_compression",
        sha256 = "51796f98a29a0d6e257c02e1f842bbc41db324758939093e6d46ec28337a3272",
        strip_prefix = "aws-c-compression-0.2.19",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/awslabs/aws-c-compression/archive/v0.2.19.tar.gz",
        ],
        build_file = Label("//third_party:aws_c_compression/aws_c_compression.BUILD.bazel"),
        cmake_name = "aws_c_compression",
        cmake_target_mapping = {
            "@aws_c_compression//:aws_c_compression": "aws_c_compression::aws_c_compression",
        },
        bazel_to_cmake = {},
    )
