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
        sha256 = "6e6bed6f75cf54006b6bafb01b3b96df19605572131a2260fddaf0e87949ced0",
        strip_prefix = "aws-checksums-0.1.5",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/awslabs/aws-checksums/archive/v0.1.5.tar.gz",
            "https://github.com/awslabs/aws-checksums/archive/v0.1.5.tar.gz",
        ],
        build_file = Label("//third_party:com_github_aws_checksums/aws_checksums.BUILD.bazel"),
        system_build_file = Label("//third_party:com_github_aws_checksums/system.BUILD.bazel"),
        cmake_name = "aws_checksums",
        cmake_target_mapping = {
            "@com_github_aws_checksums//:aws_checksums": "aws_checksums::aws_checksums",
        },
        bazel_to_cmake = {},
    )
