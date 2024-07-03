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
        name = "com_github_aws_c_http",
        sha256 = "a76ba75e59e1ac169df3ec00c0d1c453db1a4db85ee8acd3282a85ee63d6b31c",
        strip_prefix = "aws-c-http-0.8.2",
        urls = [
            "https://github.com/awslabs/aws-c-http/archive/refs/tags/v0.8.2.tar.gz",
        ],
        build_file = Label("//third_party:com_github_aws_c_http/aws_c_http.BUILD.bazel"),
        system_build_file = Label("//third_party:com_github_aws_c_http/system.BUILD.bazel"),
        cmake_name = "aws_c_http",
        cmake_target_mapping = {
            "@com_github_aws_c_http//:aws_c_http": "aws_c_http::aws_c_http",
        },
        bazel_to_cmake = {},
    )
