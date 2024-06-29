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
        name = "com_github_aws_c_event_stream",
        sha256 = "31d880d1c868d3f3df1e1f4b45e56ac73724a4dc3449d04d47fc0746f6f077b6",
        strip_prefix = "aws-c-event-stream-0.1.4",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/awslabs/aws-c-event-stream/archive/v0.1.4.tar.gz",
            "https://github.com/awslabs/aws-c-event-stream/archive/v0.1.4.tar.gz",
        ],
        build_file = Label("//third_party:com_github_aws_c_event_stream/aws_c_event_stream.BUILD.bazel"),
        system_build_file = Label("//third_party:com_github_aws_c_event_stream/system.BUILD.bazel"),
        cmake_name = "aws_c_event_stream",
        cmake_target_mapping = {
            "@com_github_aws_c_event_stream//:aws_c_event_stream": "aws_c_event_stream::aws_c_event_stream",
        },
        bazel_to_cmake = {},
    )
