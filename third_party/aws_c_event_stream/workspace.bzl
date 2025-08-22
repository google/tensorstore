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
load(
    "//third_party:repo.bzl",
    "mirror_url",
    "third_party_http_archive",
)

def repo():
    maybe(
        third_party_http_archive,
        name = "aws_c_event_stream",
        sha256 = "e94a8172e7d198d11bc7aa769c5334f1a8518f2b5bd4446d37d18fb5683623fd",
        strip_prefix = "aws-c-event-stream-0.5.6",
        urls = mirror_url("https://github.com/awslabs/aws-c-event-stream/archive/v0.5.6.tar.gz"),
        build_file = Label("//third_party:aws_c_event_stream/aws_c_event_stream.BUILD.bazel"),
        cmake_name = "aws_c_event_stream",
        cmake_target_mapping = {
            "@aws_c_event_stream//:aws_c_event_stream": "aws_c_event_stream::aws_c_event_stream",
        },
        bazel_to_cmake = {},
    )
