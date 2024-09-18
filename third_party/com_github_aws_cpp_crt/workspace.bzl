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
        name = "com_github_aws_cpp_crt",
        sha256 = "9689854b67b1a436b1cd31aae75eed8669fbb8d6240fe36684133f93e345f1ac",
        strip_prefix = "aws-crt-cpp-0.27.1",
        urls = [
            "https://github.com/awslabs/aws-crt-cpp/archive/refs/tags/v0.27.1.tar.gz",
        ],
        build_file = Label("//third_party:com_github_aws_cpp_crt/aws_cpp_crt.BUILD.bazel"),
        system_build_file = Label("//third_party:com_github_aws_cpp_crt/system.BUILD.bazel"),
        cmake_name = "aws_cpp_crt",
        cmake_target_mapping = {
            "@com_github_aws_cpp_crt//:aws_cpp_crt": "aws_cpp_crt::aws_cpp_crt",
        },
        bazel_to_cmake = {},
    )
