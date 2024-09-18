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
        name = "com_github_aws_c_iot",
        sha256 = "6b9ae985d9b019304e86e49fc6da738ed5fff3b2778ed3617db551f1e033cadf",
        strip_prefix = "aws-c-iot-0.1.21",
        urls = [
            "https://github.com/awslabs/aws-c-iot/archive/refs/tags/v0.1.21.tar.gz",
        ],
        build_file = Label("//third_party:com_github_aws_c_iot/aws_c_iot.BUILD.bazel"),
        system_build_file = Label("//third_party:com_github_aws_c_iot/system.BUILD.bazel"),
        cmake_name = "aws_c_iot",
        cmake_target_mapping = {
            "@com_github_aws_c_iot//:aws_c_iot": "aws_c_iot::aws_c_iot",
        },
        bazel_to_cmake = {},
    )
