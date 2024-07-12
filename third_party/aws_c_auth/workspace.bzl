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
        name = "aws_c_auth",
        sha256 = "f249a12a6ac319e929c005fb7efd5534c83d3af3a3a53722626ff60a494054bb",
        strip_prefix = "aws-c-auth-0.7.22",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/awslabs/aws-c-auth/archive/refs/tags/v0.7.22.tar.gz",
        ],
        build_file = Label("//third_party:aws_c_auth/aws_c_auth.BUILD.bazel"),
        cmake_name = "aws_c_auth",
        cmake_target_mapping = {
            "@aws_c_auth//:aws_c_auth": "aws_c_auth::aws_c_auth",
        },
        bazel_to_cmake = {},
    )
