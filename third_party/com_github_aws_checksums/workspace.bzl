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

load("//third_party:repo.bzl", "third_party_http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_github_aws_checksums",
        sha256 = "bdba9d0a8b8330a89c6b8cbc00b9aa14f403d3449b37ff2e0d96d62a7301b2ee",
        strip_prefix = "aws-checksums-0.1.18",
        urls = [
            "https://github.com/awslabs/aws-checksums/archive/v0.1.18.tar.gz",
        ],
        build_file = Label("//third_party:com_github_aws_checksums/aws_checksums.BUILD.bazel"),
        system_build_file = Label("//third_party:com_github_aws_checksums/system.BUILD.bazel"),
        cmake_name = "aws_checksums",
        cmake_target_mapping = {
            "@com_github_aws_checksums//:aws_checksums": "aws_checksums::aws_checksums",
        },
        bazel_to_cmake = {},
    )
