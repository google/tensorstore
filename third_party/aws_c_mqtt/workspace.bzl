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
        name = "aws_c_mqtt",
        sha256 = "63c402b8b81b107e5c1b9b6ae0065bc025b6ad4347518bf30fbd958f999e037e",
        strip_prefix = "aws-c-mqtt-0.10.1",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/awslabs/aws-c-mqtt/archive/refs/tags/v0.10.1.tar.gz",
        ],
        build_file = Label("//third_party:aws_c_mqtt/aws_c_mqtt.BUILD.bazel"),
        cmake_name = "aws_c_mqtt",
        cmake_target_mapping = {
            "@aws_c_mqtt//:aws_c_mqtt": "aws_c_mqtt::aws_c_mqtt",
        },
        bazel_to_cmake = {},
    )
