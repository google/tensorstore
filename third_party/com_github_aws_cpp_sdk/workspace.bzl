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
        name = "com_github_aws_cpp_sdk",
        patch_cmds = [
            """sed -i.bak 's/UUID::RandomUUID/Aws::Utils::UUID::RandomUUID/g' aws-cpp-sdk-core/source/client/AWSClient.cpp""",
            """sed -i.bak 's/__attribute__((visibility("default")))//g' aws-cpp-sdk-core/include/aws/core/external/tinyxml2/tinyxml2.h """,
        ],
        sha256 = "ae1cb22225b1f47eee351c0064be5e87676bf7090bb9ad19888bea0dab0e2749",
        strip_prefix = "aws-sdk-cpp-1.8.187",
        urls = [
            "https://github.com/aws/aws-sdk-cpp/archive/1.8.187.tar.gz",
        ],
        build_file = Label("//third_party:com_github_aws_cpp_sdk/aws_cpp_sdk.BUILD.bazel"),
        system_build_file = Label("//third_party:com_github_aws_cpp_sdk/system.BUILD.bazel"),
        cmake_name = "aws_cpp_sdk",
        cmake_target_mapping = {
            "@com_github_aws_cpp_sdk//:aws_cpp_sdk": "aws_cpp_sdk::aws_cpp_sdk",
        },
        bazel_to_cmake = {},
    )
